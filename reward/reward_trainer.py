"""
Reward model training with Bradley-Terry preference modeling.

This module implements the training loop for reward models using
preference data. The model learns to assign higher rewards to
preferred (chosen) responses compared to rejected ones.

Training uses the Bradley-Terry model of preferences:
P(chosen > rejected) = sigmoid(reward(chosen) - reward(rejected))
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
        PreTrainedTokenizer,
        TrainingArguments,
        get_scheduler,
    )
except ImportError:
    PreTrainedTokenizer = None
    TrainingArguments = None

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from reward.reward_model import RewardModel, RewardModelConfig


@dataclass
class RewardTrainingConfig:
    """Configuration for reward model training.

    Attributes:
        output_dir: Directory for checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Training batch size
        per_device_eval_batch_size: Evaluation batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup proportion
        lr_scheduler_type: LR scheduler type
        logging_steps: Logging interval
        save_steps: Checkpoint interval
        eval_steps: Evaluation interval
        bf16: Use bfloat16
        gradient_checkpointing: Enable gradient checkpointing
        loss_type: Loss type ("bradley_terry", "margin", "hinge")
        margin: Margin for margin-based loss
        label_smoothing: Label smoothing factor
        length_normalize: Normalize rewards by length
        max_length: Maximum sequence length
        seed: Random seed
        report_to: Logging service
    """

    output_dir: str = "./outputs/reward"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = True
    loss_type: str = "bradley_terry"
    margin: float = 0.0
    label_smoothing: float = 0.0
    length_normalize: bool = False
    max_length: int = 2048
    seed: int = 42
    report_to: str = "wandb"


class RewardTrainingCallback:
    """Callback for reward model training.

    Provides hooks for custom behavior during training.

    Args:
        on_step: Called after each training step
        on_epoch: Called after each epoch
        on_evaluate: Called after evaluation
    """

    def __init__(
        self,
        on_step: Callable | None = None,
        on_epoch: Callable | None = None,
        on_evaluate: Callable | None = None,
    ) -> None:
        """Initialize the callback.

        Args:
            on_step: Step callback function
            on_epoch: Epoch callback function
            on_evaluate: Evaluation callback function
        """
        self._on_step = on_step
        self._on_epoch = on_epoch
        self._on_evaluate = on_evaluate

    def on_train_step(
        self,
        step: int,
        loss: float,
        metrics: dict[str, float],
    ) -> None:
        """Called after each training step.

        Args:
            step: Current step number
            loss: Training loss
            metrics: Additional metrics
        """
        if self._on_step is not None:
            self._on_step(step, loss, metrics)

    def on_train_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Called after each epoch.

        Args:
            epoch: Current epoch number
            metrics: Epoch metrics
        """
        if self._on_epoch is not None:
            self._on_epoch(epoch, metrics)

    def on_evaluation(
        self,
        metrics: dict[str, float],
    ) -> None:
        """Called after evaluation.

        Args:
            metrics: Evaluation metrics
        """
        if self._on_evaluate is not None:
            self._on_evaluate(metrics)


class PreferenceDataset(TorchDataset):
    """PyTorch Dataset for preference pairs."""

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ) -> None:
        """Initialize the dataset.

        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected'
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")

        # Tokenize chosen response
        chosen_text = prompt + chosen
        chosen_encoded = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize rejected response
        rejected_text = prompt + rejected
        rejected_encoded = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_encoded["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoded["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoded["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoded["attention_mask"].squeeze(0),
            "chosen_length": len(self.tokenizer.encode(chosen)),
            "rejected_length": len(self.tokenizer.encode(rejected)),
        }


class RewardTrainer:
    """Trainer for reward models.

    Handles the complete reward model training workflow:
    - Data preparation with preference pairs
    - Bradley-Terry preference loss computation
    - Training loop with checkpointing
    - Evaluation and metric logging

    Args:
        model: RewardModel to train
        config: Training configuration
        tokenizer: Tokenizer for encoding
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        callbacks: Training callbacks
    """

    def __init__(
        self,
        model: RewardModel | None = None,
        config: RewardTrainingConfig | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        callbacks: list[RewardTrainingCallback] | None = None,
    ) -> None:
        """Initialize the reward trainer.

        Args:
            model: Reward model to train
            config: Training configuration
            tokenizer: Tokenizer
            train_dataset: Training data
            eval_dataset: Evaluation data
            callbacks: Training callbacks
        """
        self.model = model
        self.config = config or RewardTrainingConfig()
        self.tokenizer = tokenizer or (model.tokenizer if model else None)
        self.raw_train_dataset = train_dataset
        self.raw_eval_dataset = eval_dataset
        self.callbacks = callbacks or []

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
        self.scaler = None

        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup WandB logging if available."""
        self.use_wandb = (
            WANDB_AVAILABLE
            and self.config.report_to == "wandb"
            and wandb.api.api_key is not None
        )

        if self.use_wandb:
            try:
                wandb.init(
                    project="reward-model-training",
                    config=self.config.__dict__,
                )
            except Exception:
                self.use_wandb = False

    def train(self) -> dict[str, float]:
        """Run the training loop.

        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            raise ValueError("Model not set")
        if self.train_dataset is None:
            raise ValueError("Training dataset not set")

        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

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

        # Setup mixed precision
        use_amp = self.config.bf16 and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_amp else torch.float32

        # Training metrics
        training_metrics = {
            "train_loss": 0.0,
            "train_accuracy": 0.0,
            "chosen_rewards_mean": 0.0,
            "rejected_rewards_mean": 0.0,
        }

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
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    # Forward pass for chosen
                    chosen_outputs = self.model(
                        input_ids=batch["chosen_input_ids"],
                        attention_mask=batch["chosen_attention_mask"],
                    )
                    chosen_rewards = chosen_outputs["rewards"]

                    # Forward pass for rejected
                    rejected_outputs = self.model(
                        input_ids=batch["rejected_input_ids"],
                        attention_mask=batch["rejected_attention_mask"],
                    )
                    rejected_rewards = rejected_outputs["rewards"]

                    # Length normalization if enabled
                    if self.config.length_normalize:
                        chosen_rewards = chosen_rewards / batch["chosen_length"].float()
                        rejected_rewards = rejected_rewards / batch["rejected_length"].float()

                    # Compute loss
                    loss = self.compute_loss(chosen_rewards, rejected_rewards)
                    loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Accumulate metrics
                with torch.no_grad():
                    accuracy = self.compute_accuracy(chosen_rewards, rejected_rewards)
                    total_loss += loss.item() * self.config.gradient_accumulation_steps
                    total_accuracy += accuracy
                    num_batches += 1

                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / num_batches
                        avg_accuracy = total_accuracy / num_batches

                        metrics = {
                            "loss": avg_loss,
                            "accuracy": avg_accuracy,
                            "chosen_rewards_mean": chosen_rewards.mean().item(),
                            "rejected_rewards_mean": rejected_rewards.mean().item(),
                            "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
                            "lr": self.scheduler.get_last_lr()[0],
                        }

                        progress_bar.set_postfix(
                            loss=f"{avg_loss:.4f}",
                            acc=f"{avg_accuracy:.4f}",
                        )

                        if self.use_wandb:
                            wandb.log(metrics, step=self.global_step)

                        # Callbacks
                        for callback in self.callbacks:
                            callback.on_train_step(self.global_step, avg_loss, metrics)

                    # Evaluation
                    if (
                        self.eval_dataset is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        self.model.train()

                        for callback in self.callbacks:
                            callback.on_evaluation(eval_metrics)

                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_model(
                            str(self.output_dir / f"checkpoint-{self.global_step}")
                        )

            # End of epoch
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": total_loss / max(num_batches, 1),
                "train_accuracy": total_accuracy / max(num_batches, 1),
            }

            for callback in self.callbacks:
                callback.on_train_epoch(epoch, epoch_metrics)

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
            Dictionary with evaluation metrics
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
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Forward pass
                chosen_outputs = self.model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                )
                chosen_rewards = chosen_outputs["rewards"]

                rejected_outputs = self.model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"],
                )
                rejected_rewards = rejected_outputs["rewards"]

                # Compute metrics
                loss = self.compute_loss(chosen_rewards, rejected_rewards)
                accuracy = self.compute_accuracy(chosen_rewards, rejected_rewards)

                total_loss += loss.item()
                total_accuracy += accuracy
                all_chosen_rewards.append(chosen_rewards.cpu())
                all_rejected_rewards.append(rejected_rewards.cpu())
                num_batches += 1

        # Aggregate rewards
        all_chosen = torch.cat(all_chosen_rewards)
        all_rejected = torch.cat(all_rejected_rewards)

        metrics = {
            "eval_loss": total_loss / max(num_batches, 1),
            "eval_accuracy": total_accuracy / max(num_batches, 1),
            "chosen_rewards_mean": all_chosen.mean().item(),
            "chosen_rewards_std": all_chosen.std().item(),
            "rejected_rewards_mean": all_rejected.mean().item(),
            "rejected_rewards_std": all_rejected.std().item(),
            "reward_margin": (all_chosen - all_rejected).mean().item(),
        }

        if self.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.global_step)

        return metrics

    def compute_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the preference loss.

        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses

        Returns:
            Loss value
        """
        if self.config.loss_type == "bradley_terry":
            return self.compute_bradley_terry_loss(chosen_rewards, rejected_rewards)
        elif self.config.loss_type == "margin":
            return self.compute_margin_loss(
                chosen_rewards, rejected_rewards, self.config.margin
            )
        elif self.config.loss_type == "hinge":
            return self.compute_hinge_loss(chosen_rewards, rejected_rewards)
        else:
            return self.compute_bradley_terry_loss(chosen_rewards, rejected_rewards)

    def compute_bradley_terry_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Bradley-Terry preference loss.

        Loss = -log(sigmoid(chosen - rejected))

        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses

        Returns:
            Bradley-Terry loss
        """
        logits = chosen_rewards - rejected_rewards

        if self.config.margin > 0:
            logits = logits - self.config.margin

        # Label smoothing
        if self.config.label_smoothing > 0:
            labels = torch.ones_like(logits) * (1 - self.config.label_smoothing)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="mean"
            )
        else:
            # Standard Bradley-Terry: -log(sigmoid(chosen - rejected))
            loss = -F.logsigmoid(logits).mean()

        return loss

    def compute_margin_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """Compute margin-based ranking loss.

        Loss = max(0, margin - (chosen - rejected))

        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses
            margin: Margin value

        Returns:
            Margin loss
        """
        return F.relu(margin - (chosen_rewards - rejected_rewards)).mean()

    def compute_hinge_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hinge loss for preference learning.

        Args:
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses

        Returns:
            Hinge loss
        """
        margin = self.config.margin if self.config.margin > 0 else 1.0
        return F.relu(margin - (chosen_rewards - rejected_rewards)).mean()

    def prepare_dataset(
        self,
        dataset: Dataset,
    ) -> PreferenceDataset:
        """Prepare dataset for training.

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

        return PreferenceDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
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
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected

        chosen_encoded = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        rejected_encoded = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_attention_mask": chosen_encoded["attention_mask"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_attention_mask": rejected_encoded["attention_mask"],
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

        if not isinstance(dataset, PreferenceDataset):
            dataset = self.prepare_dataset(dataset)

        return DataLoader(
            dataset,
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def save_model(self, output_dir: str | None = None) -> None:
        """Save the trained model.

        Args:
            output_dir: Save directory
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save reward model
        self.model.save_pretrained(save_dir)

        # Save training config
        import json

        with open(save_dir / "training_config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save training state
        torch.save(
            {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
            },
            save_dir / "training_state.pt",
        )

    def compute_metrics(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            chosen_rewards: Rewards for chosen
            rejected_rewards: Rewards for rejected

        Returns:
            Dictionary of metrics
        """
        return {
            "accuracy": self.compute_accuracy(chosen_rewards, rejected_rewards),
            "chosen_mean": chosen_rewards.mean().item(),
            "chosen_std": chosen_rewards.std().item(),
            "rejected_mean": rejected_rewards.mean().item(),
            "rejected_std": rejected_rewards.std().item(),
            "margin": (chosen_rewards - rejected_rewards).mean().item(),
        }

    def compute_accuracy(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> float:
        """Compute preference accuracy.

        Args:
            chosen_rewards: Rewards for chosen
            rejected_rewards: Rewards for rejected

        Returns:
            Accuracy (proportion where chosen > rejected)
        """
        return (chosen_rewards > rejected_rewards).float().mean().item()

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
        model_config: RewardModelConfig | None = None,
    ) -> "RewardTrainer":
        """Create trainer from config file.

        Args:
            config_path: Path to YAML config
            model_config: Optional model config

        Returns:
            Configured RewardTrainer
        """
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract training config
        training_config = RewardTrainingConfig(**config_dict.get("training", {}))

        # Create model config if not provided
        if model_config is None:
            model_config = RewardModelConfig(**config_dict.get("model", {}))

        # Create reward model
        model = RewardModel(config=model_config)

        return cls(
            model=model,
            config=training_config,
            tokenizer=model.tokenizer,
        )
