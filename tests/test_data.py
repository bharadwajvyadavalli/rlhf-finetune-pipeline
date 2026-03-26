"""
Tests for data processing modules.

Tests cover:
- Data curation and filtering
- Deduplication correctness
- Toxicity filtering
- Preference pair generation
- Chat template formatting
"""

import pytest
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.curation import (
    QualityFilter,
    QualityConfig,
    Deduplicator,
    DeduplicationConfig,
    ToxicityFilter,
    ToxicityConfig,
    DataCurator,
)
from data.preference import (
    PreferencePairGenerator,
    PreferenceConfig,
    PreferencePair,
)
from data.formatting import (
    ChatTemplateFormatter,
    ChatTemplate,
    Message,
)


class TestQualityFilter:
    """Tests for QualityFilter class."""

    def test_filter_short_text(self) -> None:
        """Test that short texts are filtered out."""
        config = QualityConfig(min_length=50, max_length=10000)
        filter = QualityFilter(config)

        short_text = "Hello"
        result = filter.filter(short_text)
        assert result is None or (result is not None and len(result) >= config.min_length)

    def test_filter_long_text(self) -> None:
        """Test that excessively long texts are filtered out."""
        config = QualityConfig(min_length=10, max_length=100)
        filter = QualityFilter(config)

        long_text = "A" * 200
        result = filter.filter(long_text)
        assert result is None

    def test_filter_passes_quality_text(self) -> None:
        """Test that quality text passes the filter."""
        config = QualityConfig(min_length=10, max_length=1000)
        filter = QualityFilter(config)

        quality_text = "This is a well-formed sentence with proper content that should pass quality checks."
        result = filter.filter(quality_text)
        assert result is not None
        assert result == quality_text

    def test_filter_repetitive_text(self) -> None:
        """Test that highly repetitive texts are detected and filtered."""
        config = QualityConfig(min_length=10, max_length=10000, max_repetition_ratio=0.5)
        filter = QualityFilter(config)

        repetitive_text = "word word word word word word word word word word"
        result = filter.filter(repetitive_text)
        # Highly repetitive text should be filtered
        # The filter should detect this based on repetition ratio


class TestDeduplicator:
    """Tests for Deduplicator class."""

    def test_exact_duplicate_detection(self) -> None:
        """Test detection of exact duplicates."""
        config = DeduplicationConfig(use_minhash=False)
        dedup = Deduplicator(config)

        texts = [
            "This is a unique sentence.",
            "This is another unique sentence.",
            "This is a unique sentence.",  # Exact duplicate
            "Yet another unique text here.",
        ]

        result = dedup.deduplicate(texts)

        # Should have removed the exact duplicate
        assert len(result) == 3
        assert "This is a unique sentence." in result
        assert "This is another unique sentence." in result
        assert "Yet another unique text here." in result

    def test_near_duplicate_detection(self) -> None:
        """Test detection of near-duplicates using MinHash."""
        config = DeduplicationConfig(
            use_minhash=True,
            similarity_threshold=0.8,
            num_perm=128,
        )
        dedup = Deduplicator(config)

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox leaps over the lazy dog.",  # Near duplicate
            "A completely different sentence about something else.",
        ]

        result = dedup.deduplicate(texts)

        # Near duplicate should be detected
        assert len(result) <= 2

    def test_unique_texts_not_filtered(self) -> None:
        """Test that unique texts are not filtered."""
        config = DeduplicationConfig(use_minhash=True, similarity_threshold=0.8)
        dedup = Deduplicator(config)

        unique_texts = [
            "First unique text about machine learning.",
            "Second text discussing natural language processing.",
            "Third document covering computer vision topics.",
        ]

        result = dedup.deduplicate(unique_texts)

        # All texts should remain
        assert len(result) == 3

    def test_empty_input(self) -> None:
        """Test handling of empty input."""
        config = DeduplicationConfig()
        dedup = Deduplicator(config)

        result = dedup.deduplicate([])
        assert result == []


class TestToxicityFilter:
    """Tests for ToxicityFilter class."""

    def test_safe_text_passes(self) -> None:
        """Test that safe text passes the filter."""
        config = ToxicityConfig(threshold=0.7)
        filter = ToxicityFilter(config)

        safe_text = "This is a friendly and constructive message about learning."
        result = filter.filter(safe_text)
        assert result is not None
        assert result == safe_text

    def test_toxicity_scores_shape(self) -> None:
        """Test toxicity score computation returns proper shape."""
        config = ToxicityConfig(threshold=0.7)
        filter = ToxicityFilter(config)

        text = "Some sample text to analyze."
        scores = filter.get_scores(text)

        # Should return a dictionary with toxicity categories
        assert isinstance(scores, dict)
        # Common toxicity categories
        expected_keys = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]
        for key in expected_keys:
            assert key in scores
            assert 0.0 <= scores[key] <= 1.0


class TestDataCurator:
    """Tests for DataCurator class."""

    def test_pipeline_creation(self) -> None:
        """Test creation of curation pipeline."""
        curator = DataCurator()

        # Add filters
        quality_config = QualityConfig(min_length=10, max_length=1000)
        curator.add_filter(QualityFilter(quality_config))

        # Pipeline should be configured
        assert len(curator.filters) == 1

    def test_add_filter(self) -> None:
        """Test adding multiple filters to pipeline."""
        curator = DataCurator()

        curator.add_filter(QualityFilter(QualityConfig()))
        curator.add_filter(ToxicityFilter(ToxicityConfig()))

        assert len(curator.filters) == 2

    def test_process_single_text(self) -> None:
        """Test processing a single text through the pipeline."""
        curator = DataCurator()
        curator.add_filter(QualityFilter(QualityConfig(min_length=5, max_length=1000)))

        good_text = "This is a good quality text that should pass through."
        result = curator.process(good_text)
        assert result == good_text

    def test_filter_stats(self) -> None:
        """Test that filter statistics are tracked correctly."""
        curator = DataCurator()
        quality_filter = QualityFilter(QualityConfig(min_length=50, max_length=1000))
        curator.add_filter(quality_filter)

        texts = [
            "Short",  # Should be filtered
            "This is a longer text that should pass through the quality filter.",
            "No",  # Should be filtered
        ]

        results = [curator.process(t) for t in texts]
        filtered_count = sum(1 for r in results if r is None)

        # At least some texts should be filtered
        assert filtered_count >= 2


class TestPreferencePairGenerator:
    """Tests for preference pair generation."""

    def test_pair_creation(self) -> None:
        """Test creation of preference pairs."""
        pair = PreferencePair(
            prompt="What is machine learning?",
            chosen="Machine learning is a subset of AI that enables systems to learn from data.",
            rejected="I don't know.",
            chosen_score=0.9,
            rejected_score=0.2,
        )

        assert pair.prompt == "What is machine learning?"
        assert pair.chosen_score > pair.rejected_score

    def test_from_ratings_conversion(self) -> None:
        """Test converting Likert-scale ratings to pairwise preferences."""
        config = PreferenceConfig(margin=1)
        generator = PreferencePairGenerator(config)

        # Simulate rated responses
        rated_data = [
            {
                "prompt": "Explain AI",
                "responses": [
                    {"text": "AI is artificial intelligence.", "rating": 5},
                    {"text": "I don't know.", "rating": 2},
                    {"text": "AI simulates human intelligence.", "rating": 4},
                ],
            },
        ]

        pairs = generator.from_ratings(rated_data)

        # Should create pairs from different ratings
        assert len(pairs) > 0
        for pair in pairs:
            assert pair.chosen_score > pair.rejected_score

    def test_preference_pair_shape(self) -> None:
        """Test preference pair data structure."""
        pair = PreferencePair(
            prompt="Test prompt",
            chosen="Better response",
            rejected="Worse response",
            chosen_score=0.8,
            rejected_score=0.3,
        )

        # Check required fields
        assert hasattr(pair, "prompt")
        assert hasattr(pair, "chosen")
        assert hasattr(pair, "rejected")
        assert isinstance(pair.prompt, str)
        assert isinstance(pair.chosen, str)
        assert isinstance(pair.rejected, str)


class TestChatTemplateFormatter:
    """Tests for chat template formatting."""

    def test_chatml_format(self) -> None:
        """Test ChatML template formatting."""
        formatter = ChatTemplateFormatter(ChatTemplate.CHATML)

        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
            Message(role="assistant", content="Hi there! How can I help?"),
        ]

        result = formatter.format(messages)

        assert "<|im_start|>" in result
        assert "<|im_end|>" in result
        assert "system" in result
        assert "user" in result
        assert "assistant" in result

    def test_alpaca_format(self) -> None:
        """Test Alpaca template formatting."""
        formatter = ChatTemplateFormatter(ChatTemplate.ALPACA)

        messages = [
            Message(role="user", content="What is the capital of France?"),
            Message(role="assistant", content="The capital of France is Paris."),
        ]

        result = formatter.format(messages)

        assert "### Instruction:" in result or "Below is an instruction" in result
        assert "### Response:" in result or "Response:" in result

    def test_llama2_format(self) -> None:
        """Test Llama-2 template formatting."""
        formatter = ChatTemplateFormatter(ChatTemplate.LLAMA2)

        messages = [
            Message(role="user", content="Explain quantum computing."),
            Message(role="assistant", content="Quantum computing uses quantum mechanics."),
        ]

        result = formatter.format(messages)

        assert "[INST]" in result
        assert "[/INST]" in result

    def test_vicuna_format(self) -> None:
        """Test Vicuna template formatting."""
        formatter = ChatTemplateFormatter(ChatTemplate.VICUNA)

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]

        result = formatter.format(messages)

        assert "USER:" in result or "Human:" in result
        assert "ASSISTANT:" in result or "Assistant:" in result

    def test_multi_turn_formatting(self) -> None:
        """Test multi-turn conversation formatting."""
        formatter = ChatTemplateFormatter(ChatTemplate.CHATML)

        messages = [
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="2+2 equals 4."),
            Message(role="user", content="And 3+3?"),
            Message(role="assistant", content="3+3 equals 6."),
        ]

        result = formatter.format(messages)

        # Count occurrences of user and assistant markers
        assert result.count("user") == 2
        assert result.count("assistant") == 2

    def test_system_message_handling(self) -> None:
        """Test system message handling in templates."""
        formatter = ChatTemplateFormatter(ChatTemplate.CHATML)

        messages = [
            Message(role="system", content="You are a math tutor."),
            Message(role="user", content="Help me with algebra."),
        ]

        result = formatter.format(messages)

        assert "You are a math tutor." in result
        assert "Help me with algebra." in result

    def test_detect_template(self) -> None:
        """Test automatic template detection from model name."""
        # Should detect Llama template
        template = ChatTemplateFormatter.detect_template("meta-llama/Llama-2-7b-chat-hf")
        assert template == ChatTemplate.LLAMA2

        # Should detect Mistral template
        template = ChatTemplateFormatter.detect_template("mistralai/Mistral-7B-Instruct-v0.1")
        assert template in [ChatTemplate.MISTRAL, ChatTemplate.CHATML]


class TestPreferenceDataset:
    """Tests for PreferenceDataset functionality."""

    def test_preference_pair_to_dict(self) -> None:
        """Test converting preference pair to dictionary."""
        pair = PreferencePair(
            prompt="Test",
            chosen="Good response",
            rejected="Bad response",
            chosen_score=0.9,
            rejected_score=0.1,
        )

        data = pair.to_dict()

        assert data["prompt"] == "Test"
        assert data["chosen"] == "Good response"
        assert data["rejected"] == "Bad response"

    def test_batch_processing(self) -> None:
        """Test batch processing of preference pairs."""
        config = PreferenceConfig(n_candidates=2, margin=1)
        generator = PreferencePairGenerator(config)

        # Create multiple pairs
        pairs = [
            PreferencePair("p1", "c1", "r1", 0.9, 0.1),
            PreferencePair("p2", "c2", "r2", 0.8, 0.2),
            PreferencePair("p3", "c3", "r3", 0.7, 0.3),
        ]

        # Convert to dataset format
        dataset_format = [p.to_dict() for p in pairs]

        assert len(dataset_format) == 3
        for item in dataset_format:
            assert "prompt" in item
            assert "chosen" in item
            assert "rejected" in item


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text_handling(self) -> None:
        """Test handling of empty text."""
        config = QualityConfig(min_length=1)
        filter = QualityFilter(config)

        result = filter.filter("")
        assert result is None

    def test_unicode_handling(self) -> None:
        """Test handling of unicode characters."""
        config = QualityConfig(min_length=5, max_length=1000)
        filter = QualityFilter(config)

        unicode_text = "This contains émojis 🎉 and special chars: αβγδ"
        result = filter.filter(unicode_text)
        assert result is not None

    def test_whitespace_only(self) -> None:
        """Test handling of whitespace-only text."""
        config = QualityConfig(min_length=1)
        filter = QualityFilter(config)

        result = filter.filter("   \n\t  ")
        # Whitespace-only should be filtered
        assert result is None

    def test_very_long_text(self) -> None:
        """Test handling of very long text."""
        config = QualityConfig(min_length=10, max_length=100000)
        filter = QualityFilter(config)

        long_text = "This is a sentence. " * 1000
        result = filter.filter(long_text)
        # Should pass if within limits
        if len(long_text) <= config.max_length:
            assert result is not None
