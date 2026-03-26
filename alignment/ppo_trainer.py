"""
PPO (Proximal Policy Optimization) trainer for RLHF.

This module implements PPO for aligning language models with human
preferences. Key features:

- KL penalty to prevent distribution collapse
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Adaptive KL coefficient
- Integration with reward models

PPO optimizes the policy to maximize rewards while staying close to
the reference model distribution.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        GenerationConfig as HFGenerationConfig,
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

from reward.reward_model import RewardModel


@dataclass
class PPOConfig:
    """Configuration for PPO training.

    Attributes:
        output_dir: Output directory for checkpoints
        total_steps: Total number of PPO steps
        rollout_batch_size: Batch size for rollout collection
        mini_batch_size: Mini-batch size for PPO updates
        ppo_epochs: Number of PPO epochs per batch
        learning_rate: Learning rate
        init_kl_coef: Initial KL penalty coefficient
        target_kl: Target KL divergence for adaptive coefficient
        kl_penalty: KL penalty type ("kl", "abs", "mse", "full")
        gamma: Discount factor
        lam: GAE lambda parameter
        cliprange: PPO clip range for policy
        cliprange_value: Clip range for value function
        vf_coef: Value function coefficient in loss
        ent_coef: Entropy coefficient
        max_grad_norm: Maximum gradient norm
        whiten_rewards: Whether to normalize rewards
        reward_scale: Reward scaling factor
        reward_clip: Reward clipping value
        bf16: Use bfloat16
        gradient_checkpointing: Enable gradient checkpointing
        seed: Random seed
        report_to: Logging service
    """

    output_dir: str = "./outputs/ppo"
    total_steps: int = 10000
    rollout_batch_size: int = 64
    mini_batch_size: int = 8
    ppo_epochs: int = 4
    learning_rate: float = 1e-5
    init_kl_coef: float = 0.1
    target_kl: float | None = 6.0
    kl_penalty: str = "kl"
    gamma: float = 1.0
    lam: float = 0.95
    cliprange: float = 0.2
    cliprange_value: float | None = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    whiten_rewards: bool = True
    reward_scale: float = 1.0
    reward_clip: float | None = 10.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    report_to: str = "wandb"


@dataclass
class GenerationConfig:
    """Configuration for response generation.

    Attributes:
        max_new_tokens: Maximum tokens to generate
        min_new_tokens: Minimum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        do_sample: Whether to sample
    """

    max_new_tokens: int = 256
    min_new_tokens: int = 16
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    do_sample: bool = True


@dataclass
class PPOStats:
    """Statistics from a PPO training step.

    Attributes:
        loss: Total loss
        policy_loss: Policy gradient loss
        value_loss: Value function loss
        entropy: Policy entropy
        kl_div: KL divergence from reference
        mean_reward: Mean reward
        std_reward: Reward standard deviation
        mean_advantage: Mean advantage
        clipfrac: Fraction of clipped updates
        approx_kl: Approximate KL divergence
        mean_response_length: Mean response length
    """

    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_div: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_advantage: float = 0.0
    clipfrac: float = 0.0
    approx_kl: float = 0.0
    mean_response_length: float = 0.0


class ValueHead(nn.Module):
    """Value head for PPO value function estimation."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.dense(hidden_states))
        return self.out(x).squeeze(-1)


class PPOTrainer:
    """PPO trainer for RLHF.

    Implements the PPO algorithm for aligning language models using
    a reward model signal.

    Args:
        config: PPO configuration
        policy_model: Policy model to train
        ref_model: Reference model for KL penalty
        reward_model: Reward model for scoring
        tokenizer: Tokenizer for encoding/decoding
        train_dataset: Dataset with prompts
        generation_config: Generation configuration
    """

    def __init__(
        self,
        config: PPOConfig | None = None,
        policy_model: PreTrainedModel | None = None,
        ref_model: PreTrainedModel | None = None,
        reward_model: RewardModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        train_dataset: Dataset | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        """Initialize the PPO trainer.

        Args:
            config: PPO configuration
            policy_model: Policy model
            ref_model: Reference model
            reward_model: Reward model
            tokenizer: Tokenizer
            train_dataset: Training dataset
            generation_config: Generation config
        """
        self.config = config or PPOConfig()
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.generation_config = generation_config or GenerationConfig()

        # Create reference model copy if not provided
        if self.ref_model is None and self.policy_model is not None:
            self.ref_model = copy.deepcopy(self.policy_model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # Create value head
        self.value_head = None
        if self.policy_model is not None:
            hidden_size = self._get_hidden_size(self.policy_model)
            self.value_head = ValueHead(hidden_size)

        # KL coefficient
        self.kl_coef = self.config.init_kl_coef

        # Training state
        self.global_step = 0
        self.optimizer = None
        self.scheduler = None

        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _get_hidden_size(self, model: PreTrainedModel) -> int:
        """Get hidden size from model config."""
        if hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        elif hasattr(model.config, "n_embd"):
            return model.config.n_embd
        else:
            return 768

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
                        project="ppo-training",
                        config=self.config.__dict__,
                    )
            except Exception:
                self.use_wandb = False

    def train(self) -> dict[str, float]:
        """Run the PPO training loop.

        Returns:
            Dictionary with training metrics
        """
        if self.policy_model is None:
            raise ValueError("Policy model not set")
        if self.train_dataset is None:
            raise ValueError("Training dataset not set")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model.to(device)
        self.value_head.to(device)

        if self.ref_model is not None:
            self.ref_model.to(device)
        if self.reward_model is not None:
            self.reward_model.to(device)

        if self.config.gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable()

        # Setup optimizer
        params = list(self.policy_model.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate)

        # Setup scheduler
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(self.config.total_steps * 0.1),
            num_training_steps=self.config.total_steps,
        )

        # Get prompts
        prompts = list(self.train_dataset["prompt"])

        # Training metrics
        all_stats = []

        # Training loop
        self.policy_model.train()
        self.value_head.train()

        steps_per_epoch = len(prompts) // self.config.rollout_batch_size
        progress_bar = tqdm(total=self.config.total_steps, desc="PPO Training")

        while self.global_step < self.config.total_steps:
            # Sample prompts
            batch_indices = torch.randint(
                0, len(prompts), (self.config.rollout_batch_size,)
            )
            batch_prompts = [prompts[i] for i in batch_indices]

            # Perform PPO step
            stats = self.step(batch_prompts)
            all_stats.append(stats)

            self.global_step += 1
            progress_bar.update(1)

            # Logging
            if self.global_step % 10 == 0:
                progress_bar.set_postfix(
                    reward=f"{stats.mean_reward:.3f}",
                    kl=f"{stats.kl_div:.3f}",
                    loss=f"{stats.loss:.3f}",
                )

                if self.use_wandb:
                    wandb.log({
                        "reward/mean": stats.mean_reward,
                        "reward/std": stats.std_reward,
                        "kl/divergence": stats.kl_div,
                        "kl/coefficient": self.kl_coef,
                        "loss/total": stats.loss,
                        "loss/policy": stats.policy_loss,
                        "loss/value": stats.value_loss,
                        "loss/entropy": stats.entropy,
                        "train/clipfrac": stats.clipfrac,
                        "train/response_length": stats.mean_response_length,
                    }, step=self.global_step)

            # Save checkpoint
            if self.global_step % 500 == 0:
                self.save_model(str(self.output_dir / f"checkpoint-{self.global_step}"))

        progress_bar.close()

        # Final save
        self.save_model()

        if self.use_wandb:
            wandb.finish()

        # Aggregate metrics
        return {
            "mean_reward": sum(s.mean_reward for s in all_stats) / len(all_stats),
            "final_kl": all_stats[-1].kl_div if all_stats else 0.0,
            "total_steps": self.global_step,
        }

    def step(
        self,
        prompts: list[str],
    ) -> PPOStats:
        """Perform a single PPO step.

        Args:
            prompts: Batch of prompts

        Returns:
            Statistics from the step
        """
        device = next(self.policy_model.parameters()).device

        # Generate responses
        responses, response_ids = self.generate_responses(prompts)

        # Compute rewards
        rewards = self.compute_rewards(prompts, responses)

        # Process rewards
        if self.config.reward_clip is not None:
            rewards = torch.clamp(rewards, -self.config.reward_clip, self.config.reward_clip)
        rewards = rewards * self.config.reward_scale

        if self.config.whiten_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Tokenize full sequences
        full_texts = [p + r for p, r in zip(prompts, responses)]
        encoded = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Get old log probs and values
        with torch.no_grad():
            old_log_probs = self.get_log_probs(
                self.policy_model, input_ids, attention_mask, input_ids
            )
            ref_log_probs = self.get_log_probs(
                self.ref_model, input_ids, attention_mask, input_ids
            )
            old_values = self.get_values(input_ids, attention_mask)

        # Compute advantages
        advantages, returns = self.compute_advantages(rewards, old_values)

        # PPO update epochs
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0

        for _ in range(self.config.ppo_epochs):
            # Get current log probs and values
            log_probs = self.get_log_probs(
                self.policy_model, input_ids, attention_mask, input_ids
            )
            values = self.get_values(input_ids, attention_mask)

            # Compute losses
            policy_loss, clipfrac = self.compute_policy_loss(
                log_probs, old_log_probs, advantages
            )
            value_loss = self.compute_value_loss(values, old_values, returns)
            entropy = self.compute_entropy(log_probs)

            # KL penalty
            kl_div = self.compute_kl_divergence(log_probs, ref_log_probs)

            # Total loss
            loss = (
                policy_loss
                + self.config.vf_coef * value_loss
                - self.config.ent_coef * entropy
                + self.kl_coef * kl_div
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_clipfrac += clipfrac.item()

        # Update KL coefficient
        final_kl = self.compute_kl_divergence(log_probs, ref_log_probs).item()
        self.update_kl_coefficient(final_kl)

        # Compute statistics
        return PPOStats(
            loss=total_loss / self.config.ppo_epochs,
            policy_loss=total_policy_loss / self.config.ppo_epochs,
            value_loss=total_value_loss / self.config.ppo_epochs,
            entropy=total_entropy / self.config.ppo_epochs,
            kl_div=final_kl,
            mean_reward=rewards.mean().item(),
            std_reward=rewards.std().item(),
            mean_advantage=advantages.mean().item(),
            clipfrac=total_clipfrac / self.config.ppo_epochs,
            approx_kl=final_kl,
            mean_response_length=sum(len(r) for r in responses) / len(responses),
        )

    def generate_responses(
        self,
        prompts: list[str],
    ) -> tuple[list[str], torch.Tensor]:
        """Generate responses for prompts.

        Args:
            prompts: Batch of prompts

        Returns:
            Tuple of (responses, response_ids)
        """
        device = next(self.policy_model.parameters()).device

        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Generate
        self.policy_model.eval()
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **encoded,
                max_new_tokens=self.generation_config.max_new_tokens,
                min_new_tokens=self.generation_config.min_new_tokens,
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k if self.generation_config.top_k > 0 else None,
                repetition_penalty=self.generation_config.repetition_penalty,
                do_sample=self.generation_config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self.policy_model.train()

        # Decode responses (excluding prompt)
        prompt_lengths = encoded["input_ids"].shape[1]
        response_ids = outputs[:, prompt_lengths:]

        responses = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )

        return responses, response_ids

    def compute_rewards(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        """Compute rewards for prompt-response pairs.

        Args:
            prompts: Prompts
            responses: Generated responses

        Returns:
            Reward tensor
        """
        if self.reward_model is None:
            # Fallback: use response length as proxy
            return torch.tensor(
                [len(r) / 100.0 for r in responses],
                device=next(self.policy_model.parameters()).device,
            )

        return self.reward_model.compute_rewards(prompts, responses)

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns.

        Args:
            rewards: Reward values
            values: Value estimates

        Returns:
            Tuple of (advantages, returns)
        """
        # For single-step rewards, simplified GAE
        advantages = rewards - values
        returns = rewards

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute PPO policy loss with clipping.

        Args:
            log_probs: Current log probabilities
            old_log_probs: Old log probabilities
            advantages: Advantage estimates

        Returns:
            Tuple of (loss, clip_fraction)
        """
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.config.cliprange,
            1.0 + self.config.cliprange,
        )

        policy_loss_1 = -advantages * ratio
        policy_loss_2 = -advantages * clipped_ratio
        policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

        # Compute clip fraction
        clipfrac = torch.mean(
            (torch.abs(ratio - 1.0) > self.config.cliprange).float()
        )

        return policy_loss, clipfrac

    def compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value function loss with optional clipping.

        Args:
            values: Current value estimates
            old_values: Old value estimates
            returns: Computed returns

        Returns:
            Value loss
        """
        if self.config.cliprange_value is not None:
            # Clipped value loss
            values_clipped = old_values + torch.clamp(
                values - old_values,
                -self.config.cliprange_value,
                self.config.cliprange_value,
            )
            vf_loss_1 = (values - returns) ** 2
            vf_loss_2 = (values_clipped - returns) ** 2
            return 0.5 * torch.max(vf_loss_1, vf_loss_2).mean()
        else:
            return 0.5 * ((values - returns) ** 2).mean()

    def compute_entropy(
        self,
        log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy entropy.

        Args:
            log_probs: Log probabilities

        Returns:
            Entropy value
        """
        return -log_probs.mean()

    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence from reference model.

        Args:
            log_probs: Current log probabilities
            ref_log_probs: Reference log probabilities

        Returns:
            KL divergence
        """
        if self.config.kl_penalty == "kl":
            return (log_probs - ref_log_probs).mean()
        elif self.config.kl_penalty == "abs":
            return (log_probs - ref_log_probs).abs().mean()
        elif self.config.kl_penalty == "mse":
            return 0.5 * ((log_probs - ref_log_probs) ** 2).mean()
        else:
            return (log_probs - ref_log_probs).mean()

    def update_kl_coefficient(
        self,
        kl_div: float,
    ) -> None:
        """Update adaptive KL coefficient.

        Args:
            kl_div: Current KL divergence
        """
        if self.config.target_kl is None:
            return

        if kl_div > self.config.target_kl * 1.5:
            self.kl_coef *= 1.5
        elif kl_div < self.config.target_kl / 1.5:
            self.kl_coef /= 1.5

        # Clamp to reasonable range
        self.kl_coef = max(0.001, min(10.0, self.kl_coef))

    def get_log_probs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Get log probabilities from a model.

        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels

        Returns:
            Log probabilities
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
        token_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask padding
        shift_mask = attention_mask[..., 1:].contiguous()
        token_log_probs = token_log_probs * shift_mask

        # Sum per sequence
        return token_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)

    def get_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get value estimates from value head.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Value estimates
        """
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]

        # Get last token position
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, sequence_lengths]

        return self.value_head(last_hidden)

    def save_model(self, output_dir: str | None = None) -> None:
        """Save the trained policy model.

        Args:
            output_dir: Save directory
        """
        save_dir = Path(output_dir) if output_dir else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save policy model
        self.policy_model.save_pretrained(save_dir / "policy_model")
        self.tokenizer.save_pretrained(save_dir / "tokenizer")

        # Save value head
        torch.save(self.value_head.state_dict(), save_dir / "value_head.pt")

        # Save config
        import json
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save training state
        torch.save({
            "global_step": self.global_step,
            "kl_coef": self.kl_coef,
        }, save_dir / "training_state.pt")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        checkpoint_path = Path(checkpoint_path)

        # Load policy model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path / "policy_model",
            trust_remote_code=True,
        )

        # Load value head
        self.value_head.load_state_dict(
            torch.load(checkpoint_path / "value_head.pt", map_location="cpu")
        )

        # Load training state
        state = torch.load(checkpoint_path / "training_state.pt", map_location="cpu")
        self.global_step = state["global_step"]
        self.kl_coef = state["kl_coef"]

    def evaluate(
        self,
        eval_prompts: list[str],
    ) -> dict[str, float]:
        """Evaluate the current policy.

        Args:
            eval_prompts: Evaluation prompts

        Returns:
            Evaluation metrics
        """
        self.policy_model.eval()

        responses, _ = self.generate_responses(eval_prompts)
        rewards = self.compute_rewards(eval_prompts, responses)

        self.policy_model.train()

        return {
            "eval_reward_mean": rewards.mean().item(),
            "eval_reward_std": rewards.std().item(),
            "eval_response_length": sum(len(r) for r in responses) / len(responses),
        }

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path,
    ) -> "PPOTrainer":
        """Create trainer from config file.

        Args:
            config_path: Path to YAML config

        Returns:
            Configured PPOTrainer
        """
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        ppo_config = PPOConfig(**config_dict.get("ppo", {}))
        gen_config = GenerationConfig(**config_dict.get("generation", {}))

        return cls(config=ppo_config, generation_config=gen_config)
