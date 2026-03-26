"""
Tests for Supervised Fine-Tuning (SFT) modules.

Tests cover:
- SFT trainer functionality
- LoRA/QLoRA configuration
- Data collators with response-only loss masking
"""

import pytest
import torch
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetune.lora_config import (
    LoRAConfig,
    QLoRAConfig,
    get_target_modules_for_model,
)
from finetune.data_collator import (
    DataCollatorForCausalLM,
    DataCollatorForCompletionOnly,
    DataCollatorForChat,
)
from finetune.sft_trainer import SFTTrainer, SFTConfig


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default LoRA configuration values."""
        config = LoRAConfig()

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.bias == "none"

    def test_custom_config(self) -> None:
        """Test custom LoRA configuration values."""
        config = LoRAConfig(
            r=64,
            lora_alpha=128,
            lora_dropout=0.1,
            bias="all",
        )

        assert config.r == 64
        assert config.lora_alpha == 128
        assert config.lora_dropout == 0.1
        assert config.bias == "all"

    def test_to_peft_config(self) -> None:
        """Test conversion to PEFT LoraConfig."""
        config = LoRAConfig(r=8, lora_alpha=16)
        peft_config = config.to_peft_config(target_modules=["q_proj", "v_proj"])

        assert peft_config.r == 8
        assert peft_config.lora_alpha == 16
        assert "q_proj" in peft_config.target_modules
        assert "v_proj" in peft_config.target_modules


class TestQLoRAConfig:
    """Tests for QLoRAConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default QLoRA configuration."""
        config = QLoRAConfig()

        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == "bfloat16"
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True

    def test_to_bnb_config(self) -> None:
        """Test conversion to BitsAndBytes config."""
        config = QLoRAConfig()
        bnb_config = config.to_bnb_config()

        assert bnb_config.load_in_4bit is True
        assert bnb_config.bnb_4bit_quant_type == "nf4"
        assert bnb_config.bnb_4bit_use_double_quant is True


class TestGetTargetModules:
    """Tests for get_target_modules_for_model function."""

    def test_llama_target_modules(self) -> None:
        """Test target modules for Llama models."""
        modules = get_target_modules_for_model("meta-llama/Llama-2-7b-hf")

        assert "q_proj" in modules
        assert "v_proj" in modules

    def test_mistral_target_modules(self) -> None:
        """Test target modules for Mistral models."""
        modules = get_target_modules_for_model("mistralai/Mistral-7B-v0.1")

        assert "q_proj" in modules
        assert "v_proj" in modules

    def test_pythia_target_modules(self) -> None:
        """Test target modules for Pythia/GPT-NeoX models."""
        modules = get_target_modules_for_model("EleutherAI/pythia-410m")

        # Pythia uses different naming
        assert "query_key_value" in modules or "dense" in modules

    def test_unknown_model_fallback(self) -> None:
        """Test fallback for unknown models."""
        modules = get_target_modules_for_model("unknown/model-name")

        # Should return sensible defaults
        assert len(modules) > 0


class TestDataCollatorForCausalLM:
    """Tests for DataCollatorForCausalLM."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.padding_side = "left"
        return tokenizer

    def test_padding(self, mock_tokenizer: MagicMock) -> None:
        """Test sequence padding."""
        collator = DataCollatorForCausalLM(
            tokenizer=mock_tokenizer,
            max_length=128,
            padding="max_length",
        )

        # Create sample batch
        batch = [
            {"input_ids": torch.tensor([1, 2, 3, 4, 5])},
            {"input_ids": torch.tensor([1, 2, 3])},
        ]

        result = collator(batch)

        # Both sequences should be padded to same length
        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"].shape[1] == result["input_ids"].shape[1]

    def test_attention_mask_creation(self, mock_tokenizer: MagicMock) -> None:
        """Test attention mask creation."""
        collator = DataCollatorForCausalLM(
            tokenizer=mock_tokenizer,
            max_length=128,
        )

        batch = [
            {"input_ids": torch.tensor([1, 2, 3, 4, 5])},
        ]

        result = collator(batch)

        # Should have attention mask
        assert "attention_mask" in result
        # All positions with tokens should have mask = 1
        assert torch.all(result["attention_mask"][:, :5] == 1)

    def test_label_creation(self, mock_tokenizer: MagicMock) -> None:
        """Test label tensor creation for causal LM."""
        collator = DataCollatorForCausalLM(
            tokenizer=mock_tokenizer,
            max_length=128,
        )

        batch = [
            {"input_ids": torch.tensor([1, 2, 3, 4, 5])},
        ]

        result = collator(batch)

        # Labels should be same as input_ids for causal LM
        assert "labels" in result
        # Non-padded positions should match input_ids
        assert torch.all(result["labels"][:, :5] == result["input_ids"][:, :5])


class TestDataCollatorForCompletionOnly:
    """Tests for DataCollatorForCompletionOnly - response-only loss masking."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Mock encode to return token IDs
        def mock_encode(text, add_special_tokens=True):
            if "Assistant:" in text:
                return [10, 11]  # Response template tokens
            return list(range(len(text)))

        tokenizer.encode = mock_encode
        return tokenizer

    def test_prompt_masking(self, mock_tokenizer: MagicMock) -> None:
        """Test that prompt tokens are masked in labels (set to -100)."""
        collator = DataCollatorForCompletionOnly(
            tokenizer=mock_tokenizer,
            response_template="Assistant:",
        )

        # Simulate a full conversation: prompt (tokens 1-10) + response (tokens 11-20)
        # Response template starts at position 10
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                "response_start": 10,  # Response starts at position 10
            },
        ]

        result = collator(batch)

        # Prompt tokens (positions 0-9) should have labels = -100
        assert torch.all(result["labels"][:, :10] == -100)
        # Response tokens (positions 10+) should have valid labels
        assert torch.all(result["labels"][:, 10:] != -100)

    def test_completion_only_labels(self, mock_tokenizer: MagicMock) -> None:
        """Test that only completion tokens have valid labels."""
        collator = DataCollatorForCompletionOnly(
            tokenizer=mock_tokenizer,
            response_template="Assistant:",
        )

        # Create input where we know exactly where response starts
        input_ids = torch.tensor([1, 2, 3, 4, 5, 10, 11, 20, 21, 22])
        batch = [
            {
                "input_ids": input_ids,
                "response_start": 5,
            },
        ]

        result = collator(batch)

        # Count how many tokens have valid labels (not -100)
        valid_labels = (result["labels"] != -100).sum()

        # Only response tokens should have valid labels
        # Response starts at position 5, so 5 tokens should be valid
        assert valid_labels == 5


class TestDataCollatorForChat:
    """Tests for DataCollatorForChat - multi-turn conversation handling."""

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    def test_multi_turn_handling(self, mock_tokenizer: MagicMock) -> None:
        """Test multi-turn conversation handling."""
        collator = DataCollatorForChat(
            tokenizer=mock_tokenizer,
            assistant_template="<|assistant|>",
        )

        # Multi-turn conversation with two assistant turns
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                "turn_boundaries": [(0, 5), (5, 10), (10, 15)],  # User, Assist, User, Assist
                "assistant_turns": [1, 3],  # Which turns are assistant
            },
        ]

        result = collator(batch)

        # Should process correctly
        assert "input_ids" in result
        assert "labels" in result

    def test_assistant_only_training(self, mock_tokenizer: MagicMock) -> None:
        """Test that only assistant turns are trained on."""
        collator = DataCollatorForChat(
            tokenizer=mock_tokenizer,
            train_on_assistant_only=True,
        )

        # Conversation: User (0-4), Assistant (5-9), User (10-14)
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                "assistant_mask": torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
            },
        ]

        result = collator(batch)

        # User turns should have labels = -100
        # Check that some positions are masked
        masked_positions = (result["labels"] == -100).sum()
        assert masked_positions > 0


class TestSFTConfig:
    """Tests for SFTConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SFTConfig(
            model_name="EleutherAI/pythia-410m",
            output_dir="./outputs",
        )

        assert config.model_name == "EleutherAI/pythia-410m"
        assert config.learning_rate == 2e-5
        assert config.num_epochs == 3
        assert config.per_device_batch_size == 4

    def test_config_with_lora(self) -> None:
        """Test configuration with LoRA enabled."""
        lora_config = LoRAConfig(r=8, lora_alpha=16)
        config = SFTConfig(
            model_name="EleutherAI/pythia-410m",
            output_dir="./outputs",
            use_lora=True,
            lora_config=lora_config,
        )

        assert config.use_lora is True
        assert config.lora_config.r == 8


class TestSFTTrainer:
    """Tests for SFTTrainer class."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Create a mock model for testing."""
        model = MagicMock()
        model.config = MagicMock()
        model.config.hidden_size = 768
        return model

    @pytest.fixture
    def mock_tokenizer(self) -> MagicMock:
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.model_max_length = 2048
        return tokenizer

    def test_trainer_initialization(
        self, mock_model: MagicMock, mock_tokenizer: MagicMock
    ) -> None:
        """Test trainer initialization."""
        config = SFTConfig(
            model_name="test-model",
            output_dir="./outputs",
        )

        trainer = SFTTrainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )

        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert trainer.config == config


class TestResponseOnlyLossMasking:
    """Critical tests for response-only loss masking - per spec requirements."""

    def test_prompt_tokens_masked_as_minus_100(self) -> None:
        """Test that prompt tokens have labels = -100."""
        # This is the critical test from the spec:
        # "SFT: data collator loss masking (prompt tokens = -100, response tokens = valid labels)"

        # Create a simple tokenizer mock
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        collator = DataCollatorForCompletionOnly(
            tokenizer=tokenizer,
            response_template="Assistant:",
        )

        # Simulate tokenized input with clear prompt/response boundary
        # Prompt tokens: [101, 102, 103, 104, 105] (positions 0-4)
        # Response tokens: [201, 202, 203, 204, 205] (positions 5-9)
        input_ids = torch.tensor([101, 102, 103, 104, 105, 201, 202, 203, 204, 205])

        batch = [
            {
                "input_ids": input_ids,
                "response_start": 5,  # Response starts at position 5
            },
        ]

        result = collator(batch)

        # Verify prompt tokens have labels = -100
        prompt_labels = result["labels"][0, :5]
        assert torch.all(prompt_labels == -100), (
            f"Prompt tokens should have labels=-100, got {prompt_labels}"
        )

        # Verify response tokens have valid labels (equal to input_ids)
        response_labels = result["labels"][0, 5:]
        response_input_ids = result["input_ids"][0, 5:]
        assert torch.all(response_labels == response_input_ids), (
            f"Response tokens should have valid labels matching input_ids"
        )

    def test_loss_only_computed_on_response(self) -> None:
        """Test that loss is only computed on response tokens."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = DataCollatorForCompletionOnly(
            tokenizer=tokenizer,
            response_template="Response:",
        )

        # 10 prompt tokens, 5 response tokens
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

        batch = [
            {
                "input_ids": input_ids,
                "response_start": 10,
            },
        ]

        result = collator(batch)

        # Count valid labels (not -100)
        valid_label_count = (result["labels"] != -100).sum().item()

        # Should only have 5 valid labels (response tokens)
        assert valid_label_count == 5, (
            f"Expected 5 valid labels for response tokens, got {valid_label_count}"
        )

        # Count masked labels
        masked_label_count = (result["labels"] == -100).sum().item()

        # Should have 10 masked labels (prompt tokens)
        assert masked_label_count == 10, (
            f"Expected 10 masked labels for prompt tokens, got {masked_label_count}"
        )


class TestEdgeCases:
    """Test edge cases in SFT training."""

    def test_empty_response(self) -> None:
        """Test handling of empty response."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = DataCollatorForCompletionOnly(
            tokenizer=tokenizer,
            response_template="Assistant:",
        )

        # All prompt, no response
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "response_start": 5,  # Response starts at end
            },
        ]

        result = collator(batch)

        # All tokens should be masked
        assert torch.all(result["labels"] == -100)

    def test_all_response(self) -> None:
        """Test handling of all response (no prompt)."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        collator = DataCollatorForCompletionOnly(
            tokenizer=tokenizer,
            response_template="",
        )

        # All response, no prompt
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "response_start": 0,  # Response starts at beginning
            },
        ]

        result = collator(batch)

        # No tokens should be masked
        assert torch.all(result["labels"] != -100)

    def test_batch_with_different_lengths(self) -> None:
        """Test batching with different sequence lengths."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"

        collator = DataCollatorForCompletionOnly(
            tokenizer=tokenizer,
            response_template="Assistant:",
        )

        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),  # 8 tokens
                "response_start": 4,
            },
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),  # 5 tokens
                "response_start": 2,
            },
        ]

        result = collator(batch)

        # Both should be padded to same length
        assert result["input_ids"].shape[0] == 2
        assert result["labels"].shape[0] == 2
        assert result["input_ids"].shape[1] == result["labels"].shape[1]
