"""
Reward model architecture for RLHF.

This module implements the reward model, which consists of a base language
model with a scalar reward head. The model takes a prompt-response pair
and outputs a scalar reward score indicating quality.

The reward model is trained on preference data (chosen/rejected pairs)
using Bradley-Terry preference modeling.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
except ImportError:
    PreTrainedModel = None
    PreTrainedTokenizer = None


@dataclass
class RewardModelConfig:
    """Configuration for the reward model.

    Attributes:
        base_model_name: Name/path of the base language model
        hidden_dim: Hidden dimension for reward head MLP (None = use model dim)
        num_layers: Number of hidden layers in reward head
        dropout: Dropout probability in reward head
        activation: Activation function ("relu", "gelu", "silu")
        init_near_zero: Initialize output to near-zero values
        freeze_base_layers: Number of base model layers to freeze
        use_last_token: Use last token representation for reward
        use_mean_pooling: Use mean pooling over sequence
        normalize_rewards: Whether to normalize reward outputs
        load_in_4bit: Use 4-bit quantization
        load_in_8bit: Use 8-bit quantization
    """

    base_model_name: str = "meta-llama/Llama-2-7b-hf"
    hidden_dim: int | None = None
    num_layers: int = 1
    dropout: float = 0.1
    activation: str = "relu"
    init_near_zero: bool = True
    freeze_base_layers: int = 0
    use_last_token: bool = True
    use_mean_pooling: bool = False
    normalize_rewards: bool = False
    load_in_4bit: bool = True
    load_in_8bit: bool = False


class RewardHead(nn.Module):
    """Scalar reward head that produces a single reward value.

    Takes the hidden representation from the base model and produces
    a scalar reward through an MLP.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Hidden dimension (None = no hidden layer)
        num_layers: Number of hidden layers
        dropout: Dropout probability
        activation: Activation function name
        init_near_zero: Initialize to output near-zero values
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        init_near_zero: bool = True,
    ) -> None:
        """Initialize the reward head.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            activation: Activation function
            init_near_zero: Initialize for near-zero output
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.activation_name = activation
        self.init_near_zero = init_near_zero

        layers: list[nn.Module] = []
        current_dim = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                # Output layer (scalar)
                layers.append(nn.Linear(current_dim, 1))
            else:
                # Hidden layer
                layers.append(nn.Linear(current_dim, self.hidden_dim))
                layers.append(self._get_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                current_dim = self.hidden_dim

        self.layers = nn.Sequential(*layers)

        if init_near_zero:
            self._init_weights()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute reward from hidden states.

        Args:
            hidden_states: Shape (batch_size, hidden_dim)

        Returns:
            Reward scores of shape (batch_size,)
        """
        output = self.layers(hidden_states)
        return output.squeeze(-1)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name.

        Args:
            name: Activation function name

        Returns:
            Activation module
        """
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def _init_weights(self) -> None:
        """Initialize weights for near-zero output."""
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize output layer to produce near-zero outputs
        if isinstance(self.layers[-1], nn.Linear):
            nn.init.normal_(self.layers[-1].weight, mean=0.0, std=0.01)
            if self.layers[-1].bias is not None:
                nn.init.zeros_(self.layers[-1].bias)


class RewardModel(nn.Module):
    """Reward model for RLHF.

    Combines a base language model with a reward head to produce
    scalar reward scores for input sequences.

    Args:
        config: RewardModelConfig with model settings
        base_model: Optional pre-loaded base model
        tokenizer: Optional pre-loaded tokenizer
    """

    def __init__(
        self,
        config: RewardModelConfig | None = None,
        base_model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ) -> None:
        """Initialize the reward model.

        Args:
            config: Model configuration
            base_model: Pre-loaded base model
            tokenizer: Pre-loaded tokenizer
        """
        super().__init__()
        self.config = config or RewardModelConfig()

        # Load or use provided base model
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = self.load_base_model()

        # Load or use provided tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Get hidden size from base model config
        hidden_size = self._get_hidden_size()

        # Remove the language model head if it exists
        self._remove_lm_head()

        # Create reward head
        self.reward_head = RewardHead(
            input_dim=hidden_size,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            activation=self.config.activation,
            init_near_zero=self.config.init_near_zero,
        )

        # Freeze specified layers
        if self.config.freeze_base_layers > 0:
            self.freeze_layers(self.config.freeze_base_layers)

    def _get_hidden_size(self) -> int:
        """Get the hidden size from the base model config."""
        model_config = self.base_model.config
        if hasattr(model_config, "hidden_size"):
            return model_config.hidden_size
        elif hasattr(model_config, "n_embd"):
            return model_config.n_embd
        elif hasattr(model_config, "d_model"):
            return model_config.d_model
        else:
            raise ValueError("Could not determine hidden size from model config")

    def _remove_lm_head(self) -> None:
        """Remove the language model head from the base model."""
        # Common attribute names for LM heads
        head_names = ["lm_head", "embed_out", "output"]
        for name in head_names:
            if hasattr(self.base_model, name):
                # Replace with identity to save memory
                setattr(self.base_model, name, nn.Identity())
                break

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute reward for input sequences.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            return_hidden_states: Whether to return intermediate states

        Returns:
            Dictionary with "rewards" and optionally "hidden_states"
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]

        # Get sequence representation
        sequence_repr = self.get_sequence_representation(hidden_states, attention_mask)

        # Compute rewards
        rewards = self.reward_head(sequence_repr)

        # Optionally normalize rewards
        if self.config.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        result = {"rewards": rewards}
        if return_hidden_states:
            result["hidden_states"] = hidden_states

        return result

    def compute_rewards(
        self,
        prompts: list[str],
        responses: list[str],
    ) -> torch.Tensor:
        """Compute rewards for prompt-response pairs.

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            Reward tensor of shape (batch_size,)
        """
        # Combine prompts and responses
        texts = [p + r for p, r in zip(prompts, responses)]

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=2048,
        )

        # Move to model device
        device = next(self.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward pass
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        return outputs["rewards"]

    def get_sequence_representation(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get sequence representation for reward computation.

        Args:
            hidden_states: Model hidden states (batch, seq, hidden)
            attention_mask: Attention mask

        Returns:
            Sequence representation (batch, hidden)
        """
        if self.config.use_mean_pooling:
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / sum_mask

        elif self.config.use_last_token:
            # Get last non-padded token position for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(
                hidden_states.size(0), device=hidden_states.device
            )
            return hidden_states[batch_indices, sequence_lengths]

        else:
            # Default: use first token (CLS-style)
            return hidden_states[:, 0, :]

    def load_base_model(self) -> PreTrainedModel:
        """Load the base language model.

        Returns:
            Loaded base model
        """
        quantization_config = None
        device_map = "auto"

        if self.config.load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                quantization_config = None
        elif self.config.load_in_8bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            except Exception:
                quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not quantization_config else None,
        )

        return model

    def freeze_layers(self, num_layers: int) -> None:
        """Freeze the bottom N layers of the base model.

        Args:
            num_layers: Number of layers to freeze
        """
        # Get transformer layers
        layers = None
        if hasattr(self.base_model, "model"):
            if hasattr(self.base_model.model, "layers"):
                layers = self.base_model.model.layers
            elif hasattr(self.base_model.model, "h"):
                layers = self.base_model.model.h
        elif hasattr(self.base_model, "transformer"):
            if hasattr(self.base_model.transformer, "h"):
                layers = self.base_model.transformer.h

        if layers is None:
            return

        # Freeze embeddings
        if hasattr(self.base_model, "model"):
            if hasattr(self.base_model.model, "embed_tokens"):
                for param in self.base_model.model.embed_tokens.parameters():
                    param.requires_grad = False
        elif hasattr(self.base_model, "transformer"):
            if hasattr(self.base_model.transformer, "wte"):
                for param in self.base_model.transformer.wte.parameters():
                    param.requires_grad = False
            if hasattr(self.base_model.transformer, "wpe"):
                for param in self.base_model.transformer.wpe.parameters():
                    param.requires_grad = False

        # Freeze specified number of layers
        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def save_pretrained(self, save_path: str | Path) -> None:
        """Save the reward model.

        Args:
            save_path: Path to save to
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        import json

        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Save reward head
        torch.save(self.reward_head.state_dict(), save_path / "reward_head.pt")

        # Save base model if not quantized
        try:
            self.base_model.save_pretrained(save_path / "base_model")
        except Exception:
            # For quantized models, just save the config
            self.base_model.config.save_pretrained(save_path / "base_model")

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path / "tokenizer")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        config: RewardModelConfig | None = None,
    ) -> "RewardModel":
        """Load a saved reward model.

        Args:
            model_path: Path to saved model
            config: Optional override config

        Returns:
            Loaded RewardModel
        """
        import json

        model_path = Path(model_path)

        # Load config
        if config is None:
            with open(model_path / "config.json", "r") as f:
                config_dict = json.load(f)
            config = RewardModelConfig(**config_dict)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path / "tokenizer", trust_remote_code=True
        )

        # Load base model
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path / "base_model",
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception:
            # Load from original model name
            base_model = None

        # Create reward model
        reward_model = cls(config=config, base_model=base_model, tokenizer=tokenizer)

        # Load reward head weights
        reward_head_path = model_path / "reward_head.pt"
        if reward_head_path.exists():
            reward_model.reward_head.load_state_dict(
                torch.load(reward_head_path, map_location="cpu")
            )

        return reward_model

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on the base model."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable()

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embeddings from base model.

        Returns:
            Input embedding layer
        """
        return self.base_model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare inputs for generation (if needed).

        Args:
            input_ids: Input token IDs
            **kwargs: Additional arguments

        Returns:
            Prepared inputs dictionary
        """
        return {"input_ids": input_ids, **kwargs}


def create_reward_model(
    base_model_name: str,
    config: RewardModelConfig | None = None,
    **kwargs: Any,
) -> RewardModel:
    """Factory function to create a reward model.

    Args:
        base_model_name: Name/path of base model
        config: Optional configuration
        **kwargs: Override config values

    Returns:
        Configured RewardModel
    """
    if config is None:
        config = RewardModelConfig(base_model_name=base_model_name)
    else:
        config.base_model_name = base_model_name

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return RewardModel(config=config)


def load_reward_model_for_inference(
    model_path: str | Path,
    device: str = "cuda",
) -> tuple[RewardModel, PreTrainedTokenizer]:
    """Load a reward model for inference.

    Args:
        model_path: Path to saved model
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    reward_model = RewardModel.from_pretrained(model_path)

    if device != "auto":
        reward_model = reward_model.to(device)

    reward_model.eval()

    return reward_model, reward_model.tokenizer
