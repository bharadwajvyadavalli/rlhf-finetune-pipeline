"""
Supervised Fine-Tuning (SFT) trainer with LoRA/QLoRA support.

This module implements the SFT stage of the RLHF pipeline, transforming
base language models into instruction-following models. Key features:

- Parameter-efficient fine-tuning with LoRA/QLoRA
- Support for multiple chat templates
- Multi-turn conversation handling
- Gradient checkpointing for memory efficiency
- Integration with Weights & Biases for experiment tracking
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Lazy imports
_TRANSFORMERS_AVAILABLE = False
_TRL_AVAILABLE = False
_PEFT_AVAILABLE = False
_WANDB_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        TrainingArguments,
        Trainer,
        TrainerCallback,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    from trl import SFTTrainer as TRLSFTTrainer
    from trl import DataCollatorForCompletionOnlyLM
    _TRL_AVAILABLE = True
except ImportError:
    pass

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    _PEFT_AVAILABLE = True
except ImportError:
    pass

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    pass

from .lora_config import (
    LoRAConfig,
    QLoRAConfig,
    get_target_modules_for_model,
    print_trainable_parameters,
)
from .data_collator import (
    DataCollatorForCompletionOnly,
    get_response_template_for_model,
)


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning.

    Attributes:
        model_name_or_path: Base model to fine-tune
        output_dir: Directory for checkpoints and outputs
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        warmup_ratio: Proportion of warmup steps
        lr_scheduler_type: Learning rate scheduler type
        max_seq_length: Maximum sequence length
        logging_steps: Steps between logging
        save_steps: Steps between checkpoints
        eval_steps: Steps between evaluations
        bf16: Use bfloat16 mixed precision
        fp16: Use float16 mixed precision
        gradient_checkpointing: Enable gradient checkpointing
        use_lora: Whether to use LoRA
        use_qlora: Whether to use QLoRA (4-bit)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: LoRA target modules
        template_type: Chat template type
        seed: Random seed
        report_to: Experiment tracking service
    """

    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
    output_dir: str = "./outputs/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    use_lora: bool = True
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    template_type: str = "alpaca"
    seed: int = 42
    report_to: str = "wandb"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    save_strategy: str = "steps"
    evaluation_strategy: str = "steps"
    optim: str = "adamw_torch"
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    packing: bool = False
    dataset_text_field: str = "text"


class SFTTrainer:
    """Supervised Fine-Tuning Trainer with LoRA/QLoRA support.

    This trainer handles the complete SFT workflow including:
    - Model loading with optional quantization
    - LoRA/QLoRA adapter setup
    - Dataset preparation and formatting
    - Training loop with checkpointing
    - Evaluation and metric logging
    """

    def __init__(
        self,
        config: SFTConfig | None = None,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Initialize the SFT Trainer.

        Args:
            config: Training configuration
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            train_dataset: Training dataset (optional)
            eval_dataset: Evaluation dataset (optional)
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for SFTTrainer")

        self.config = config or SFTConfig()
        self._model = model
        self._tokenizer = tokenizer
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        self._trainer: TRLSFTTrainer | Trainer | None = None
        self._data_collator = None

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Set seed for reproducibility
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def train(self) -> dict[str, float]:
        """Run the training loop.

        Returns:
            Dictionary with training metrics
        """
        # Load model and tokenizer if not provided
        if self._model is None:
            self._model = self.load_model()

        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()

        # Setup LoRA if enabled
        if self.config.use_lora or self.config.use_qlora:
            self._model = self.setup_lora(self._model)
            print_trainable_parameters(self._model)

        # Prepare datasets
        if self._train_dataset is not None:
            self._train_dataset = self.prepare_dataset(self._train_dataset)

        if self._eval_dataset is not None:
            self._eval_dataset = self.prepare_dataset(self._eval_dataset, is_eval=True)

        # Create data collator
        self._data_collator = self.create_data_collator()

        # Create training arguments
        training_args = self.create_training_arguments()

        # Create trainer
        if _TRL_AVAILABLE and not self.config.packing:
            # Use TRL's SFTTrainer
            self._trainer = TRLSFTTrainer(
                model=self._model,
                tokenizer=self._tokenizer,
                train_dataset=self._train_dataset,
                eval_dataset=self._eval_dataset,
                args=training_args,
                data_collator=self._data_collator,
                max_seq_length=self.config.max_seq_length,
                dataset_text_field=self.config.dataset_text_field,
                packing=self.config.packing,
            )
        else:
            # Use standard HuggingFace Trainer
            self._trainer = Trainer(
                model=self._model,
                tokenizer=self._tokenizer,
                train_dataset=self._train_dataset,
                eval_dataset=self._eval_dataset,
                args=training_args,
                data_collator=self._data_collator,
                compute_metrics=self.compute_metrics,
            )

        # Log to WandB if available
        if self.config.report_to == "wandb" and _WANDB_AVAILABLE:
            if wandb.run is None:
                wandb.init(
                    project="rlhf-sft",
                    name=f"sft-{self.config.model_name_or_path.split('/')[-1]}",
                    config=vars(self.config),
                )

        # Train
        train_result = self._trainer.train()

        # Save final model
        self.save_model()

        # Log metrics
        metrics = train_result.metrics
        self._trainer.log_metrics("train", metrics)
        self._trainer.save_metrics("train", metrics)

        return metrics

    def evaluate(self, eval_dataset: Dataset | None = None) -> dict[str, float]:
        """Evaluate the model on the evaluation dataset.

        Args:
            eval_dataset: Optional dataset to evaluate on

        Returns:
            Dictionary with evaluation metrics
        """
        if self._trainer is None:
            raise ValueError("Trainer not initialized. Call train() first.")

        dataset = eval_dataset or self._eval_dataset

        if dataset is None:
            raise ValueError("No evaluation dataset provided")

        dataset = self.prepare_dataset(dataset, is_eval=True)
        metrics = self._trainer.evaluate(eval_dataset=dataset)

        self._trainer.log_metrics("eval", metrics)
        return metrics

    def save_model(self, output_dir: str | None = None) -> None:
        """Save the model and tokenizer.

        Args:
            output_dir: Directory to save to (defaults to config.output_dir)
        """
        save_dir = output_dir or self.config.output_dir

        if self._trainer is not None:
            self._trainer.save_model(save_dir)
        elif self._model is not None:
            self._model.save_pretrained(save_dir)

        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(save_dir)

    def load_model(self) -> PreTrainedModel:
        """Load the base model with optional quantization.

        Returns:
            Loaded model
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required")

        # Prepare quantization config if using QLoRA
        quantization_config = None
        if self.config.use_qlora:
            qlora_config = QLoRAConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
            quantization_config = qlora_config.to_bnb_config()

        # Determine compute dtype
        if self.config.bf16:
            torch_dtype = torch.bfloat16
        elif self.config.fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for k-bit training if using QLoRA
        if self.config.use_qlora and _PEFT_AVAILABLE:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )
        elif self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load and configure the tokenizer.

        Returns:
            Configured tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Set padding side (left for causal LM)
        tokenizer.padding_side = "right"

        return tokenizer

    def setup_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply LoRA/QLoRA adapters to the model.

        Args:
            model: Base model

        Returns:
            Model with LoRA adapters
        """
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        # Get target modules
        target_modules = self.config.lora_target_modules
        if not target_modules or target_modules == ["auto"]:
            target_modules = get_target_modules_for_model(self.config.model_name_or_path)

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        peft_config = lora_config.to_peft_config()
        model = get_peft_model(model, peft_config)

        return model

    def prepare_dataset(
        self,
        dataset: Dataset,
        is_eval: bool = False,
    ) -> Dataset:
        """Prepare dataset for training.

        Args:
            dataset: Raw dataset
            is_eval: Whether this is for evaluation

        Returns:
            Processed dataset
        """
        # Check if dataset needs formatting
        if self.config.dataset_text_field not in dataset.column_names:
            # Format the dataset
            dataset = dataset.map(
                self.format_example,
                remove_columns=[c for c in dataset.column_names
                               if c not in [self.config.dataset_text_field]],
            )

        # Tokenize if not using TRL's SFTTrainer (which handles tokenization)
        if not _TRL_AVAILABLE:
            dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=[self.config.dataset_text_field],
            )

        return dataset

    def format_example(self, example: dict[str, Any]) -> dict[str, str]:
        """Format a single example using the chat template.

        Args:
            example: Raw example dictionary

        Returns:
            Formatted example with 'text' field
        """
        # Import formatting module
        from ..data.formatting import ChatTemplateFormatter

        formatter = ChatTemplateFormatter.get_template_for_model(
            self.config.model_name_or_path
        )

        # Handle different dataset formats
        if "instruction" in example:
            # Alpaca-style format
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")

            formatted = formatter.format_instruction(instruction, input_text, output)

        elif "messages" in example:
            # Chat format
            formatted = formatter.format(example["messages"], add_generation_prompt=False)

        elif "prompt" in example and "response" in example:
            # Prompt-response format
            formatted = formatter.format_instruction(
                example["prompt"],
                response=example["response"],
            )

        elif "text" in example:
            # Already formatted
            formatted = example["text"]

        else:
            # Try to construct from available fields
            text_parts = []
            for key in ["prompt", "question", "input", "context"]:
                if key in example:
                    text_parts.append(str(example[key]))
            for key in ["response", "answer", "output", "completion"]:
                if key in example:
                    text_parts.append(str(example[key]))

            formatted = "\n\n".join(text_parts)

        return {self.config.dataset_text_field: formatted}

    def tokenize_function(
        self, examples: dict[str, list]
    ) -> dict[str, list]:
        """Tokenize a batch of examples.

        Args:
            examples: Batch of examples

        Returns:
            Tokenized batch
        """
        return self._tokenizer(
            examples[self.config.dataset_text_field],
            truncation=True,
            max_length=self.config.max_seq_length,
            padding=False,
            return_tensors=None,
        )

    def create_data_collator(self) -> Any:
        """Create the appropriate data collator.

        Returns:
            Data collator instance
        """
        if _TRL_AVAILABLE:
            # Get response template for completion-only training
            response_template = get_response_template_for_model(
                self.config.model_name_or_path
            )

            return DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=self._tokenizer,
            )
        else:
            return DataCollatorForCompletionOnly(
                tokenizer=self._tokenizer,
                response_template=get_response_template_for_model(
                    self.config.model_name_or_path
                ),
                max_length=self.config.max_seq_length,
            )

    def create_training_arguments(self) -> TrainingArguments:
        """Create HuggingFace TrainingArguments.

        Returns:
            Configured TrainingArguments
        """
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if self._eval_dataset else None,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end if self._eval_dataset else False,
            save_strategy=self.config.save_strategy,
            evaluation_strategy=self.config.evaluation_strategy if self._eval_dataset else "no",
            optim=self.config.optim,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to=self.config.report_to if _WANDB_AVAILABLE else "none",
            seed=self.config.seed,
        )

    def compute_metrics(
        self, eval_preds: tuple[Any, Any]
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            eval_preds: Tuple of predictions and labels

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_preds

        # Shift predictions and labels for causal LM
        # predictions are shifted right, labels are shifted left
        predictions = predictions[:, :-1]
        labels = labels[:, 1:]

        # Flatten
        predictions = predictions.reshape(-1)
        labels = labels.reshape(-1)

        # Mask padding
        mask = labels != -100

        # Calculate accuracy
        correct = (predictions == labels) & mask
        accuracy = correct.sum() / mask.sum()

        return {"accuracy": accuracy.item()}

    def get_train_dataloader(self) -> DataLoader:
        """Get the training DataLoader.

        Returns:
            Training DataLoader
        """
        if self._trainer is None:
            raise ValueError("Trainer not initialized")

        return self._trainer.get_train_dataloader()

    def get_eval_dataloader(
        self, eval_dataset: Dataset | None = None
    ) -> DataLoader:
        """Get the evaluation DataLoader.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Evaluation DataLoader
        """
        if self._trainer is None:
            raise ValueError("Trainer not initialized")

        return self._trainer.get_eval_dataloader(eval_dataset)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload model",
        private: bool = True,
    ) -> None:
        """Push model to HuggingFace Hub.

        Args:
            repo_id: Repository ID on the Hub
            commit_message: Commit message
            private: Whether the repo should be private
        """
        if self._trainer is not None:
            self._trainer.push_to_hub(
                repo_id=repo_id,
                commit_message=commit_message,
                private=private,
            )
        elif self._model is not None:
            self._model.push_to_hub(
                repo_id=repo_id,
                commit_message=commit_message,
                private=private,
            )
            if self._tokenizer is not None:
                self._tokenizer.push_to_hub(repo_id=repo_id, private=private)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        config: SFTConfig | None = None,
    ) -> "SFTTrainer":
        """Load a trainer from a saved checkpoint.

        Args:
            model_path: Path to saved model
            config: Optional new config

        Returns:
            Loaded SFTTrainer
        """
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required")

        # Load config if exists
        config_path = Path(model_path) / "sft_config.yaml"
        if config_path.exists() and config is None:
            config = cls._load_config(config_path)

        config = config or SFTConfig(model_name_or_path=str(model_path))

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return cls(config=config, model=model, tokenizer=tokenizer)

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> "SFTTrainer":
        """Create trainer from a YAML config file.

        Args:
            config_path: Path to config YAML

        Returns:
            Configured SFTTrainer
        """
        config = cls._load_config(config_path)
        return cls(config=config)

    @staticmethod
    def _load_config(config_path: str | Path) -> SFTConfig:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return SFTConfig(**config_dict)

    def merge_and_save(self, output_path: str) -> None:
        """Merge LoRA weights and save the full model.

        Args:
            output_path: Path to save merged model
        """
        from .lora_config import merge_lora_weights

        if self._model is None:
            raise ValueError("Model not loaded")

        merged_model = merge_lora_weights(self._model, output_path)

        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(output_path)

        return merged_model
