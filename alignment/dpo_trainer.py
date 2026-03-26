"""
DPO (Direct Preference Optimization) trainer.

This module implements DPO as an alternative to PPO for alignment.
DPO directly optimizes the policy on preference data without needing
an explicit reward model during training.

Key features:
- Multiple loss variants (sigmoid, hinge, IPO)
- Reference model management
- Simpler training loop than PPO
- Competitive results with less hyperparameter tuning
"""

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
        get_scheduler,
    )
except ImportError:
    PreTrainedModel = None
    PreTrainedTokenizer = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DPOLossType(Enum):
    """Types of DPO loss functions."""

    SIGMOID = "sigmoid"
    HINGE = "hinge"
    IPO = "ipo"
    KTO = "kto"


@dataclass
class DPOConfig:
    """Configuration for DPO training.

    Attributes:
        output_dir: Output directory
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size
        per_device_eval_batch_size: Evaluation batch size
        gradient_accumulation_steps: Gradient accumulation
        learning_rate: Learning rate
        beta: DPO beta parameter (KL penalty strength)
        loss_type: Loss function type
        label_smoothing: Label smoothing factor
        weight_decay: Weight decay
        warmup_ratio: Warmup proportion
        lr_scheduler_type: LR scheduler type
        max_seq_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        logging_steps: Logging interval
        save_steps: Checkpoint interval
        eval_steps: Evaluation interval
        bf16: Use bfloat16
        gradient_checkpointing: Enable gradient checkpointing
        sync_ref_model: Whether to sync ref model
        ref_model_sync_steps: Steps between ref syncs
        seed: Random seed
        report_to: Logging service
    """

    output_dir: str = "./outputs/dpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    beta: float = 0.1
    loss_type: DPOLossType = DPOLossType.SIGMOID
    label_smoothing: float = 0.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    max_prompt_length: int = 512
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = True
    sync_ref_model: bool = False
    ref_model_sync_steps: int = 512
    seed: int = 42
    report_to: str = "wandb"


class DPODataset(TorchDataset):
    """Dataset for DPO training with preference pairs."""

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 2048,
        max_prompt_length: int = 512,
    ) -> None:
        """Initialize the dataset.

        Args:
            data: List of dicts with prompt, chosen, rejected
            tokenizer: Tokenizer for encoding
            max_seq_length: Maximum total sequence length
            max_prompt_length: Maximum prompt length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        # Build full sequences
        chosen_seq = prompt + chosen
        rejected_seq = prompt + rejected

        # Tokenize
        chosen_tokens = self.tokenizer(
            chosen_seq,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        rejected_tokens = self.tokenizer(
            rejected_seq,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Get prompt length for masking
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        # Create labels (mask prompt tokens with -100)
        chosen_labels = chosen_tokens["input_ids"].clone()
        chosen_labels[:, :prompt_length] = -100

        rejected_labels = rejected_tokens["input_ids"].clone()
        rejected_labels[:, :prompt_length] = -100

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels.squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels.squeeze(0),
        }


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Implements DPO for aligning language models directly on
    preference data without an explicit reward model.

    Args:
        config: DPO configuration
        model: Policy model to train
        ref_model: Reference model (frozen)
        tokenizer: Tokenizer
        train_dataset: Training dataset with preferences
        eval_dataset: Evaluation dataset
    """

    def __init__(
        self,
        config: DPOConfig | None = None,
        model: PreTrainedModel | None = None,
        ref_model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Initialize the DPO trainer.

        Args:
            config: DPO configuration
            model: Policy model
            ref_model: Reference model
            tokenizer: Tokenizer
            train_dataset: Training data
            eval_dataset: Evaluation data
        """
        self.config = config or DPOConfig()
        self.model = model
        self.tokenizer = tokenizer
        self.raw_train_dataset = train_dataset
        self.raw_eval_dataset = eval_dataset

        # Create reference model copy if not provided
        if ref_model is None and model is not None:
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            self.ref_model = ref_model

        # Prepare datasets
        self.train_dataset = None
        self.eval_dataset = None
        if train_dataset is not None:
            self.train_dataset = self.prepare_dataset(train_dataset)
        if eval_dataset is not None:
            self.eval_dataset = self.prepare_dataset(eval_dataset)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.optimizer = None
        self.scheduler = None

        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup WandB logging if available."""
        self.use_wandb = (
            WANDB_AVAILABLE
            and self.config.report_to == "wandb"
        )

        if self.use_wandb:
            try:
                if wandb.api.api_key:
                    wandb.init(
                        project="dpo-training",
                        config=self.config.__dict__,
                    )
            except Exception:
                self.use_wandb = False

    def train(self) -> dict[str, float]:
        """Run the DPO training loop.

        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            raise ValueError("Model not set")
        if self.train_dataset is None:
            raise ValueError("Training dataset not set")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if self.ref_model is not None:
            self.ref_model.to(device)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Get dataloader
        train_dataloader = self.get_train_dataloader()

        # Calculate total steps
        num_update_steps_per_epoch = (
            len(train_dataloader) // self.config.gradient_accumulation_steps
        )
        total_steps = num_update_steps_per_epoch * self.config.num_train_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Setup scheduler
        self.scheduler = get_scheduler(
            self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training metrics
        training_metrics = {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
        }

        # Mixed precision
        use_amp = self.config.bf16 and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_amp else torch.float32

        # Training loop
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for epoch in range(self.config.num_train_epochs):
            self.current_epoch = epoch

            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}",
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    # Get policy log probs
                    policy_chosen_logps = self.get_batch_logps(
                        self.model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_labels"],
                    )
                    policy_rejected_logps = self.get_batch_logps(
                        self.model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["rejected_labels"],
                    )

                    # Get reference log probs
                    with torch.no_grad():
                        ref_chosen_logps = self.get_batch_logps(
                            self.ref_model,
                            batch["chosen_input_ids"],
                            batch["chosen_attention_mask"],
                            batch["chosen_labels"],
                        )
                        ref_rejected_logps = self.get_batch_logps(
                            self.ref_model,
                            batch["rejected_input_ids"],
                            batch["rejected_attention_mask"],
                            batch["rejected_labels"],
                        )

                    # Compute loss
                    loss, metrics = self.compute_loss(
                        policy_chosen_logps,
                        policy_rejected_logps,
                        ref_chosen_logps,
                        ref_rejected_logps,
                    )
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Accumulate metrics
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                total_accuracy += metrics["accuracy"]
                num_batches += 1

                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Sync reference model if enabled
                    if (
                        self.config.sync_ref_model
                        and self.global_step % self.config.ref_model_sync_steps == 0
                    ):
                        self.sync_reference_model()

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / num_batches
                        avg_accuracy = total_accuracy / num_batches

                        progress_bar.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            acc=f"{avg_accuracy:.4f}",
                        )

                        if self.use_wandb:
                            wandb.log({
                                "loss": avg_loss,
                                "accuracy": avg_accuracy,
                                "chosen_rewards": metrics.get("chosen_rewards", 0),
                                "rejected_rewards": metrics.get("rejected_rewards", 0),
                                "reward_margin": metrics.get("reward_margin", 0),
                                "lr": self.scheduler.get_last_lr()[0],
                            }, step=self.global_step)

                    # Evaluation
                    if (
                        self.eval_dataset is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        self.model.train()

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_model(
                            str(self.output_dir / f"checkpoint-{self.global_step}")
                        )

            # Reset counters
            total_loss = 0.0
            total_accuracy = 0.0
            num_batches = 0

        # Save final model
        self.save_model()

        # Final evaluation
        if self.eval_dataset is not None:
            training_metrics = self.evaluate()

        if self.use_wandb:
            wandb.finish()

        return training_metrics

    def evaluate(
        self, eval_dataset: Dataset | None = None
    ) -> dict[str, float]:
        """Evaluate the model.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Evaluation metrics
        """
        if eval_dataset is not None:
            self.eval_dataset = self.prepare_dataset(eval_dataset)

        if self.eval_dataset is None:
            raise ValueError("No evaluation dataset provided")

        self.model.eval()
        device = next(self.model.parameters()).device

        eval_dataloader = self.get_eval_dataloader()

        total_loss = 0.0
        total_accuracy = 0.0
        all_chosen_rewards = []
        all_rejected_rewards = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}

                # Get log probs
                policy_chosen_logps = self.get_batch_logps(
                    self.model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                policy_rejected_logps = self.get_batch_logps(
                    self.model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )
                ref_chosen_logps = self.get_batch_logps(
                    self.ref_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                ref_rejected_logps = self.get_batch_logps(
                    self.ref_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )

                # Compute loss and metrics
                loss, metrics = self.compute_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                )

                total_loss += loss.item()
                total_accuracy += metrics["accuracy"]
                all_chosen_rewards.append(metrics["chosen_rewards"])
                all_rejected_rewards.append(metrics["rejected_rewards"])
                num_batches += 1

        metrics = {
            "eval_loss": total_loss / max(num_batches, 1),
            "eval_accuracy": total_accuracy / max(num_batches, 1),
            "eval_chosen_rewards": sum(all_chosen_rewards) / max(num_batches, 1),
            "eval_rejected_rewards": sum(all_rejected_rewards) / max(num_batches, 1),
        }

        if self.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.global_step)

        return metrics

    def compute_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DPO loss.

        Args:
            policy_chosen_logps: Policy log probs for chosen
            policy_rejected_logps: Policy log probs for rejected
            ref_chosen_logps: Reference log probs for chosen
            ref_rejected_logps: Reference log probs for rejected

        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Compute log ratios
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        # Compute loss based on type
        loss_type = self.config.loss_type
        if isinstance(loss_type, str):
            loss_type = DPOLossType(loss_type)

        if loss_type == DPOLossType.SIGMOID:
            loss = self.compute_sigmoid_loss(chosen_logratios, rejected_logratios)
        elif loss_type == DPOLossType.HINGE:
            loss = self.compute_hinge_loss(chosen_logratios, rejected_logratios)
        elif loss_type == DPOLossType.IPO:
            loss = self.compute_ipo_loss(chosen_logratios, rejected_logratios)
        else:
            loss = self.compute_sigmoid_loss(chosen_logratios, rejected_logratios)

        # Compute metrics
        with torch.no_grad():
            accuracy = self.compute_accuracy(chosen_logratios, rejected_logratios)
            chosen_rewards = self.config.beta * chosen_logratios.mean().item()
            rejected_rewards = self.config.beta * rejected_logratios.mean().item()

        metrics = {
            "accuracy": accuracy,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "reward_margin": chosen_rewards - rejected_rewards,
        }

        return loss, metrics

    def compute_sigmoid_loss(
        self,
        chosen_logratios: torch.Tensor,
        rejected_logratios: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sigmoid (standard) DPO loss.

        Loss = -log(sigmoid(beta * (chosen_logratio - rejected_logratio)))

        Args:
            chosen_logratios: Log ratios for chosen
            rejected_logratios: Log ratios for rejected

        Returns:
            Sigmoid loss
        """
        logits = self.config.beta * (chosen_logratios - rejected_logratios)

        if self.config.label_smoothing > 0:
            # Label smoothing
            labels = torch.ones_like(logits) * (1 - self.config.label_smoothing)
            return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        else:
            return -F.logsigmoid(logits).mean()

    def compute_hinge_loss(
        self,
        chosen_logratios: torch.Tensor,
        rejected_logratios: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hinge DPO loss.

        More robust to noise in preferences.

        Args:
            chosen_logratios: Log ratios for chosen
            rejected_logratios: Log ratios for rejected

        Returns:
            Hinge loss
        """
        logits = self.config.beta * (chosen_logratios - rejected_logratios)
        return F.relu(1.0 - logits).mean()

    def compute_ipo_loss(
        self,
        chosen_logratios: torch.Tensor,
        rejected_logratios: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IPO (Identity Preference Optimization) loss.

        Uses different regularization approach.

        Args:
            chosen_logratios: Log ratios for chosen
            rejected_logratios: Log ratios for rejected

        Returns:
            IPO loss
        """
        logits = chosen_logratios - rejected_logratios
        # IPO uses squared hinge-like loss
        return ((logits - 1.0 / (2 * self.config.beta)) ** 2).mean()

    def get_batch_logps(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for a batch.

        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels

        Returns:
            Per-sequence log probabilities
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Create mask for non-ignored tokens
        loss_mask = (shift_labels != -100).float()

        # Replace -100 with 0 for gathering
        labels_for_gather = shift_labels.clone()
        labels_for_gather[labels_for_gather == -100] = 0

        # Gather log probs for labels
        token_log_probs = torch.gather(
            log_probs, -1, labels_for_gather.unsqueeze(-1)
        ).squeeze(-1)

        # Apply mask
        token_log_probs = token_log_probs * loss_mask

        # Sum per sequence
        return token_log_probs.sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1)

    def prepare_dataset(
        self,
        dataset: Dataset,
    ) -> DPODataset:
        """Prepare dataset for DPO training.

        Args:
            dataset: Raw preference dataset

        Returns:
            Processed dataset
        """
        # Convert HF Dataset to list of dicts
        if hasattr(dataset, "to_list"):
            data = dataset.to_list()
        elif hasattr(dataset, "__iter__"):
            data = list(dataset)
        else:
            data = [dataset[i] for i in range(len(dataset))]

        return DPODataset(
            data=data,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            max_prompt_length=self.config.max_prompt_length,
        )

    def tokenize_pair(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> dict[str, torch.Tensor]:
        """Tokenize a preference pair.

        Args:
            prompt: Input prompt
            chosen: Chosen response
            rejected: Rejected response

        Returns:
            Tokenized inputs
        """
        chosen_seq = prompt + chosen
        rejected_seq = prompt + rejected

        chosen_tokens = self.tokenizer(
            chosen_seq,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        rejected_tokens = self.tokenizer(
            rejected_seq,
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }

    def get_train_dataloader(self) -> DataLoader:
        """Get training DataLoader.

        Returns:
            Training DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def get_eval_dataloader(
        self, eval_dataset: Dataset | None = None
    ) -> DataLoader:
        """Get evaluation DataLoader.

        Args:
            eval_dataset: Optional evaluation dataset

        Returns:
            Evaluation DataLoader
        """
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if dataset is None:
            raise ValueError("No evaluation dataset available")

        if not isinstance(dataset, DPODataset):
            dataset = self.prepare_dataset(dataset)

        return DataLoader(
            dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def sync_reference_model(self) -> None:
        """Sync reference model weights from policy model."""
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def save_model(self, output_dir: str | None = None) -> None:
        """Save the trained model.

        Args:
            output_dir: Save directory
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_dir / "model")
        self.tokenizer.save_pretrained(save_dir / "tokenizer")

        # Save config
        import json
        config_dict = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.config.__dict__.items()
        }
        with open(save_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Save training state
        torch.save({
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }, save_dir / "training_state.pt")

    def compute_metrics(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            policy_chosen_logps: Policy log probs for chosen
            policy_rejected_logps: Policy log probs for rejected
            ref_chosen_logps: Reference log probs for chosen
            ref_rejected_logps: Reference log probs for rejected

        Returns:
            Dictionary of metrics
        """
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        return {
            "accuracy": self.compute_accuracy(chosen_logratios, rejected_logratios),
            "chosen_rewards": self.config.beta * chosen_logratios.mean().item(),
            "rejected_rewards": self.config.beta * rejected_logratios.mean().item(),
        }

    def compute_accuracy(
        self,
        chosen_logratios: torch.Tensor,
        rejected_logratios: torch.Tensor,
    ) -> float:
        """Compute preference accuracy.

        Args:
            chosen_logratios: Log ratios for chosen
            rejected_logratios: Log ratios for rejected

        Returns:
            Accuracy (proportion where chosen > rejected)
        """
        return (chosen_logratios > rejected_logratios).float().mean().item()

    def generate_samples(
        self,
        prompts: list[str],
        num_samples: int = 1,
    ) -> list[str]:
        """Generate samples for qualitative evaluation.

        Args:
            prompts: Input prompts
            num_samples: Samples per prompt

        Returns:
            Generated responses
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        all_responses = []

        for prompt in prompts:
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).to(device)

            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **encoded,
                        max_new_tokens=256,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                response = self.tokenizer.decode(
                    outputs[0][encoded["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                all_responses.append(response)

        self.model.train()
        return all_responses

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
    ) -> "DPOTrainer":
        """Create trainer from config file.

        Args:
            config_path: Path to YAML config

        Returns:
            Configured DPOTrainer
        """
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Handle loss type enum
        if "loss_type" in config_dict:
            config_dict["loss_type"] = DPOLossType(config_dict["loss_type"])

        dpo_config = DPOConfig(**config_dict)

        return cls(config=dpo_config)
