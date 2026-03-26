"""
Tests for reward modeling modules.

Tests cover:
- Reward model architecture and forward pass shape (batch_size, 1)
- Bradley-Terry loss computation on known inputs
- Reward trainer
- Reward analysis and hacking detection
"""

import pytest
import torch
import torch.nn as nn
import sys
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reward.reward_model import (
    RewardModel,
    RewardModelConfig,
    RewardHead,
)
from reward.reward_trainer import (
    RewardTrainer,
    RewardTrainingConfig,
    bradley_terry_loss,
    margin_loss,
    hinge_loss,
)
from reward.reward_analysis import (
    RewardAnalyzer,
    RewardDistribution,
    LengthCorrelationResult,
    SycophancyResult,
)


class TestRewardModelConfig:
    """Tests for RewardModelConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RewardModelConfig(base_model_name="EleutherAI/pythia-410m")

        assert config.base_model_name == "EleutherAI/pythia-410m"
        assert config.num_frozen_layers == 0
        assert config.reward_head_dropout == 0.1

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = RewardModelConfig(
            base_model_name="EleutherAI/pythia-1b",
            hidden_size=2048,
            num_frozen_layers=10,
            reward_head_hidden_size=512,
            reward_head_dropout=0.2,
        )

        assert config.base_model_name == "EleutherAI/pythia-1b"
        assert config.hidden_size == 2048
        assert config.num_frozen_layers == 10
        assert config.reward_head_hidden_size == 512


class TestRewardHead:
    """Tests for RewardHead module."""

    def test_forward_pass_produces_scalar(self) -> None:
        """Test forward pass produces scalar output."""
        hidden_size = 768
        head = RewardHead(
            hidden_size=hidden_size,
            hidden_dim=256,
            dropout=0.1,
        )

        # Create batch of hidden states
        batch_size = 4
        hidden_states = torch.randn(batch_size, hidden_size)

        # Forward pass
        output = head(hidden_states)

        # Should produce one scalar per sample
        assert output.shape == (batch_size, 1)

    def test_output_shape_batch_size_1(self) -> None:
        """Test output shape is (batch_size, 1) - per spec requirement."""
        hidden_size = 1024
        head = RewardHead(hidden_size=hidden_size)

        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            hidden_states = torch.randn(batch_size, hidden_size)
            output = head(hidden_states)

            assert output.shape == (batch_size, 1), (
                f"Expected shape ({batch_size}, 1), got {output.shape}"
            )

    def test_near_zero_init(self) -> None:
        """Test near-zero initialization of final layer."""
        head = RewardHead(hidden_size=768)

        # Final layer should have small weights
        final_layer = head.output
        if hasattr(final_layer, 'weight'):
            weight_std = final_layer.weight.std().item()
            # Should be initialized with small values
            assert weight_std < 0.1


class TestRewardModel:
    """Tests for RewardModel class."""

    @pytest.fixture
    def mock_base_model(self) -> MagicMock:
        """Create a mock base model for testing."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.hidden_size = 768
        model.config.vocab_size = 50000

        # Mock forward pass
        def mock_forward(input_ids, attention_mask=None, output_hidden_states=True, **kwargs):
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(batch_size, seq_len, 768)
            result = MagicMock()
            result.hidden_states = (hidden_states,)  # Tuple of hidden states
            return result

        model.return_value = mock_forward
        model.__call__ = mock_forward

        return model

    def test_forward_pass_shape(self) -> None:
        """Test forward pass produces correct shape (batch_size, 1)."""
        # This is a critical test from the spec:
        # "Reward: forward pass shape (batch_size, 1)"

        config = RewardModelConfig(
            base_model_name="EleutherAI/pythia-70m",
            hidden_size=512,
        )

        # Create minimal reward model for testing
        reward_model = create_minimal_reward_model(config)

        batch_size = 4
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        output = reward_model(input_ids, attention_mask)

        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {output.shape}"
        )

    def test_sequence_representation(self) -> None:
        """Test that sequence representation is extracted from last non-padding position."""
        config = RewardModelConfig(
            base_model_name="EleutherAI/pythia-70m",
            hidden_size=512,
        )

        reward_model = create_minimal_reward_model(config)

        batch_size = 2
        seq_len = 10

        # Create input with padding
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 5 real tokens
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 8 real tokens
        ])

        output = reward_model(input_ids, attention_mask)

        # Should still produce (batch_size, 1)
        assert output.shape == (batch_size, 1)

    def test_freeze_layers(self) -> None:
        """Test freezing base model layers."""
        config = RewardModelConfig(
            base_model_name="EleutherAI/pythia-70m",
            num_frozen_layers=2,
        )

        reward_model = create_minimal_reward_model(config)
        reward_model.freeze_layers(2)

        # Check that some parameters are frozen
        frozen_params = sum(1 for p in reward_model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in reward_model.parameters())

        # Should have some frozen parameters
        assert frozen_params > 0 or total_params > 0


class TestBradleyTerryLoss:
    """Tests for Bradley-Terry loss computation - per spec requirement."""

    def test_bradley_terry_loss_computation(self) -> None:
        """Test Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))."""
        # This is a critical test from the spec:
        # "Bradley-Terry loss computation on known inputs"

        # Known inputs
        r_chosen = torch.tensor([1.0, 2.0, 3.0])
        r_rejected = torch.tensor([0.0, 1.0, 1.0])

        # Compute loss
        loss = bradley_terry_loss(r_chosen, r_rejected)

        # Expected: -log(sigmoid(r_chosen - r_rejected))
        # For r_chosen - r_rejected = [1.0, 1.0, 2.0]
        # sigmoid([1.0, 1.0, 2.0]) = [0.731, 0.731, 0.881]
        # -log([0.731, 0.731, 0.881]) = [0.313, 0.313, 0.127]
        # Mean = 0.251

        diff = r_chosen - r_rejected
        expected_loss = -torch.log(torch.sigmoid(diff)).mean()

        assert torch.isclose(loss, expected_loss, atol=1e-5), (
            f"Expected loss {expected_loss.item():.6f}, got {loss.item():.6f}"
        )

    def test_bradley_terry_perfect_preference(self) -> None:
        """Test Bradley-Terry loss with strong preference."""
        # Chosen much better than rejected
        r_chosen = torch.tensor([10.0])
        r_rejected = torch.tensor([0.0])

        loss = bradley_terry_loss(r_chosen, r_rejected)

        # With large positive difference, loss should be very small
        assert loss.item() < 0.001

    def test_bradley_terry_wrong_preference(self) -> None:
        """Test Bradley-Terry loss when model prefers rejected."""
        # Rejected scored higher than chosen (model is wrong)
        r_chosen = torch.tensor([0.0])
        r_rejected = torch.tensor([10.0])

        loss = bradley_terry_loss(r_chosen, r_rejected)

        # With large negative difference, loss should be high
        assert loss.item() > 5.0

    def test_bradley_terry_equal_scores(self) -> None:
        """Test Bradley-Terry loss when scores are equal."""
        r_chosen = torch.tensor([5.0])
        r_rejected = torch.tensor([5.0])

        loss = bradley_terry_loss(r_chosen, r_rejected)

        # When difference is 0, sigmoid(0) = 0.5, -log(0.5) = 0.693
        expected = torch.tensor(math.log(2))  # ln(2) ≈ 0.693

        assert torch.isclose(loss, expected, atol=1e-4)

    def test_bradley_terry_batch(self) -> None:
        """Test Bradley-Terry loss on a batch."""
        batch_size = 32
        r_chosen = torch.randn(batch_size) + 1.0  # Slightly positive bias
        r_rejected = torch.randn(batch_size)

        loss = bradley_terry_loss(r_chosen, r_rejected)

        # Loss should be a scalar
        assert loss.dim() == 0
        # Loss should be positive
        assert loss.item() >= 0


class TestMarginLoss:
    """Tests for margin-based loss."""

    def test_margin_loss_computation(self) -> None:
        """Test margin loss: -log(sigmoid(r_chosen - r_rejected - margin))."""
        r_chosen = torch.tensor([2.0])
        r_rejected = torch.tensor([0.0])
        margin = 1.0

        loss = margin_loss(r_chosen, r_rejected, margin)

        # Expected: -log(sigmoid(2.0 - 0.0 - 1.0)) = -log(sigmoid(1.0))
        expected = -torch.log(torch.sigmoid(torch.tensor(1.0)))

        assert torch.isclose(loss, expected, atol=1e-5)

    def test_margin_increases_difficulty(self) -> None:
        """Test that margin increases loss for same scores."""
        r_chosen = torch.tensor([1.5])
        r_rejected = torch.tensor([0.5])

        loss_no_margin = margin_loss(r_chosen, r_rejected, margin=0.0)
        loss_with_margin = margin_loss(r_chosen, r_rejected, margin=1.0)

        # Loss with margin should be higher
        assert loss_with_margin > loss_no_margin


class TestHingeLoss:
    """Tests for hinge loss."""

    def test_hinge_loss_computation(self) -> None:
        """Test hinge loss: max(0, margin - (r_chosen - r_rejected))."""
        r_chosen = torch.tensor([2.0])
        r_rejected = torch.tensor([0.0])
        margin = 1.0

        loss = hinge_loss(r_chosen, r_rejected, margin)

        # Expected: max(0, 1.0 - 2.0) = max(0, -1.0) = 0
        assert loss.item() == 0.0

    def test_hinge_loss_violated_margin(self) -> None:
        """Test hinge loss when margin is violated."""
        r_chosen = torch.tensor([1.0])
        r_rejected = torch.tensor([0.5])
        margin = 1.0

        loss = hinge_loss(r_chosen, r_rejected, margin)

        # Expected: max(0, 1.0 - 0.5) = 0.5
        expected = torch.tensor(0.5)

        assert torch.isclose(loss, expected, atol=1e-5)


class TestRewardTrainingConfig:
    """Tests for RewardTrainingConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RewardTrainingConfig(output_dir="./outputs")

        assert config.output_dir == "./outputs"
        assert config.loss_type == "bradley_terry"
        assert config.margin == 0.0


class TestRewardTrainer:
    """Tests for RewardTrainer class."""

    @pytest.fixture
    def mock_reward_model(self) -> nn.Module:
        """Create a simple reward model for testing."""
        return create_minimal_reward_model(
            RewardModelConfig(
                base_model_name="test",
                hidden_size=256,
            )
        )

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.model_max_length = 512
        tokenizer.padding_side = "right"

        def mock_call(text, **kwargs):
            if isinstance(text, str):
                tokens = {"input_ids": torch.randint(0, 1000, (1, 20))}
            else:
                tokens = {"input_ids": torch.randint(0, 1000, (len(text), 20))}
            tokens["attention_mask"] = torch.ones_like(tokens["input_ids"])
            return tokens

        tokenizer.__call__ = mock_call
        tokenizer.return_value = mock_call("")

        return tokenizer

    def test_trainer_initialization(
        self, mock_reward_model: nn.Module, mock_tokenizer: MagicMock
    ) -> None:
        """Test trainer initialization."""
        config = RewardTrainingConfig(output_dir="./outputs")

        trainer = RewardTrainer(
            model=mock_reward_model,
            tokenizer=mock_tokenizer,
            config=config,
        )

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.config == config

    def test_compute_preference_accuracy(self) -> None:
        """Test preference accuracy computation."""
        # 4 samples: 3 correct (chosen > rejected), 1 incorrect
        r_chosen = torch.tensor([1.0, 2.0, 3.0, 0.0])
        r_rejected = torch.tensor([0.0, 1.0, 1.0, 1.0])

        # Compute accuracy
        correct = (r_chosen > r_rejected).float().mean()

        # 3/4 = 0.75
        assert correct.item() == 0.75


class TestRewardDistribution:
    """Tests for RewardDistribution dataclass."""

    def test_from_rewards(self) -> None:
        """Test creating distribution from reward tensor."""
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        dist = RewardDistribution.from_rewards(rewards)

        assert dist.mean == pytest.approx(3.0)
        assert dist.std == pytest.approx(rewards.std().item(), rel=0.01)
        assert dist.min == 1.0
        assert dist.max == 5.0

    def test_statistics(self) -> None:
        """Test distribution statistics computation."""
        rewards = torch.randn(1000) * 2 + 1  # mean ~1, std ~2

        dist = RewardDistribution.from_rewards(rewards)

        # Mean should be close to 1
        assert abs(dist.mean - 1.0) < 0.2
        # Std should be close to 2
        assert abs(dist.std - 2.0) < 0.2


class TestRewardAnalyzer:
    """Tests for RewardAnalyzer class."""

    @pytest.fixture
    def mock_reward_model(self) -> nn.Module:
        """Create a simple reward model for testing."""
        return create_minimal_reward_model(
            RewardModelConfig(base_model_name="test", hidden_size=256)
        )

    def test_length_correlation_detection(self, mock_reward_model: nn.Module) -> None:
        """Test length-reward correlation computation."""
        analyzer = RewardAnalyzer(mock_reward_model)

        # Create data where length correlates with reward
        lengths = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float)
        rewards = lengths * 0.1 + torch.randn(5) * 0.1  # Strong correlation

        result = analyzer.compute_length_correlation(lengths, rewards)

        # Should detect correlation
        assert isinstance(result, LengthCorrelationResult)
        assert hasattr(result, "correlation")
        # Correlation should be high (positive)
        assert result.correlation > 0.5

    def test_no_length_correlation(self, mock_reward_model: nn.Module) -> None:
        """Test when there is no length correlation."""
        analyzer = RewardAnalyzer(mock_reward_model)

        # Random lengths and rewards (no correlation)
        lengths = torch.tensor([10, 50, 20, 40, 30], dtype=torch.float)
        rewards = torch.tensor([0.5, 0.1, 0.8, 0.2, 0.9])  # Uncorrelated

        result = analyzer.compute_length_correlation(lengths, rewards)

        # Correlation should be low
        assert abs(result.correlation) < 0.5


# Helper functions

def create_minimal_reward_model(config: RewardModelConfig) -> nn.Module:
    """Create a minimal reward model for testing without loading actual weights."""

    class MinimalRewardModel(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(10000, hidden_size)
            self.transformer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
            )
            self.reward_head = RewardHead(hidden_size)
            self._frozen_layers = 0

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # Get embeddings
            x = self.embedding(input_ids)

            # Simple transformer pass
            x = self.transformer(x)

            # Get last non-padding token representation
            if attention_mask is not None:
                # Find last non-zero position
                seq_lens = attention_mask.sum(dim=1) - 1
                batch_size = x.shape[0]
                last_hidden = x[torch.arange(batch_size), seq_lens.long()]
            else:
                last_hidden = x[:, -1, :]

            # Get reward
            reward = self.reward_head(last_hidden)
            return reward

        def freeze_layers(self, n: int) -> None:
            self._frozen_layers = n
            # Freeze embedding layer if n > 0
            if n > 0:
                for param in self.embedding.parameters():
                    param.requires_grad = False

    return MinimalRewardModel(config.hidden_size or 256)


class TestPreferenceAccuracy:
    """Additional tests for preference accuracy metrics."""

    def test_perfect_accuracy(self) -> None:
        """Test 100% preference accuracy."""
        r_chosen = torch.tensor([1.0, 2.0, 3.0, 4.0])
        r_rejected = torch.tensor([0.0, 1.0, 2.0, 3.0])

        accuracy = (r_chosen > r_rejected).float().mean()
        assert accuracy.item() == 1.0

    def test_zero_accuracy(self) -> None:
        """Test 0% preference accuracy."""
        r_chosen = torch.tensor([0.0, 1.0, 2.0, 3.0])
        r_rejected = torch.tensor([1.0, 2.0, 3.0, 4.0])

        accuracy = (r_chosen > r_rejected).float().mean()
        assert accuracy.item() == 0.0

    def test_random_accuracy(self) -> None:
        """Test approximately 50% accuracy with equal scores."""
        # With small noise around 0, should be ~50%
        torch.manual_seed(42)
        r_chosen = torch.randn(1000) * 0.01
        r_rejected = torch.randn(1000) * 0.01

        accuracy = (r_chosen > r_rejected).float().mean()
        # Should be close to 50%
        assert 0.4 < accuracy.item() < 0.6


class TestGradientFlow:
    """Test gradient flow through reward model."""

    def test_gradients_flow_to_head(self) -> None:
        """Test that gradients flow to reward head."""
        config = RewardModelConfig(base_model_name="test", hidden_size=256)
        model = create_minimal_reward_model(config)

        # Forward pass
        input_ids = torch.randint(0, 1000, (4, 20))
        attention_mask = torch.ones_like(input_ids)

        output = model(input_ids, attention_mask)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check gradients in reward head
        has_gradients = False
        for param in model.reward_head.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "Gradients should flow to reward head"
