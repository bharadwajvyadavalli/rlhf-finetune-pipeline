"""
Data curation module for quality filtering, deduplication, and toxicity removal.

This module provides tools for preprocessing raw training data to ensure only
high-quality, diverse, and safe content enters the training pipeline. Key features:

- Quality filtering based on perplexity, coherence, and heuristic rules
- Near-duplicate detection using MinHash LSH
- Toxicity filtering using classifier-based detection (detoxify)

The curation pipeline is designed to be modular, allowing different filtering
strategies to be combined as needed.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator

import torch
from datasets import Dataset
from tqdm import tqdm

# Lazy imports for optional dependencies
_DATASKETCH_AVAILABLE = False
_DETOXIFY_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False

try:
    from datasketch import MinHash, MinHashLSH
    _DATASKETCH_AVAILABLE = True
except ImportError:
    pass

try:
    from detoxify import Detoxify
    _DETOXIFY_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


class FilterType(Enum):
    """Types of data filters available."""

    QUALITY = "quality"
    DEDUP = "deduplication"
    TOXICITY = "toxicity"
    PII = "pii"
    LENGTH = "length"
    LANGUAGE = "language"


@dataclass
class FilterStats:
    """Statistics from a filtering operation.

    Attributes:
        total_samples: Total number of samples processed
        passed_samples: Number of samples that passed the filter
        removed_samples: Number of samples removed by the filter
        removal_reasons: Dictionary mapping removal reasons to counts
        filter_type: Type of filter that generated these stats
    """

    total_samples: int = 0
    passed_samples: int = 0
    removed_samples: int = 0
    removal_reasons: dict[str, int] = field(default_factory=dict)
    filter_type: FilterType | None = None

    def update(self, passed: bool, reason: str | None = None) -> None:
        """Update statistics with a new sample result."""
        self.total_samples += 1
        if passed:
            self.passed_samples += 1
        else:
            self.removed_samples += 1
            if reason:
                self.removal_reasons[reason] = self.removal_reasons.get(reason, 0) + 1

    def merge(self, other: "FilterStats") -> "FilterStats":
        """Merge two FilterStats objects."""
        merged_reasons = dict(self.removal_reasons)
        for reason, count in other.removal_reasons.items():
            merged_reasons[reason] = merged_reasons.get(reason, 0) + count

        return FilterStats(
            total_samples=self.total_samples + other.total_samples,
            passed_samples=self.passed_samples + other.passed_samples,
            removed_samples=self.removed_samples + other.removed_samples,
            removal_reasons=merged_reasons,
            filter_type=self.filter_type,
        )


@dataclass
class QualityConfig:
    """Configuration for quality filtering.

    Attributes:
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters
        min_words: Minimum number of words
        max_perplexity: Maximum perplexity score (requires perplexity model)
        min_coherence: Minimum coherence score
        remove_repetitive: Whether to remove texts with excessive repetition
        repetition_threshold: Threshold for repetition detection (0-1)
        require_punctuation: Whether to require proper punctuation
        language: Required language code (e.g., "en")
        perplexity_model_name: Model name for perplexity calculation
    """

    min_length: int = 50
    max_length: int = 100000
    min_words: int = 10
    max_perplexity: float | None = None
    min_coherence: float | None = None
    remove_repetitive: bool = True
    repetition_threshold: float = 0.3
    require_punctuation: bool = True
    language: str | None = None
    perplexity_model_name: str = "gpt2"


class BaseFilter(ABC):
    """Abstract base class for data filters.

    All filters should implement the filter method which takes a sample
    and returns a tuple of (should_keep, removal_reason).
    """

    def __init__(self) -> None:
        self.stats = FilterStats()

    @abstractmethod
    def filter(self, sample: dict[str, Any]) -> tuple[bool, str | None]:
        """Filter a single sample.

        Args:
            sample: Dictionary containing the sample data

        Returns:
            Tuple of (should_keep, removal_reason). removal_reason is None
            if should_keep is True.
        """
        pass

    def filter_batch(
        self, samples: list[dict[str, Any]]
    ) -> list[tuple[bool, str | None]]:
        """Filter a batch of samples.

        Args:
            samples: List of sample dictionaries

        Returns:
            List of (should_keep, removal_reason) tuples
        """
        return [self.filter(sample) for sample in samples]

    def reset_stats(self) -> None:
        """Reset filter statistics."""
        self.stats = FilterStats(filter_type=self.stats.filter_type)


class QualityFilter(BaseFilter):
    """Filter for data quality based on heuristic rules and model-based scores.

    This filter removes low-quality samples based on length, perplexity,
    coherence, repetition, and other quality metrics.

    Args:
        config: QualityConfig with filter parameters
        text_column: Name of the column containing the text to filter
        perplexity_model: Optional model for computing perplexity
    """

    def __init__(
        self,
        config: QualityConfig | None = None,
        text_column: str = "text",
        perplexity_model: Any | None = None,
    ) -> None:
        """Initialize the quality filter.

        Args:
            config: Quality filter configuration
            text_column: Column name containing text to filter
            perplexity_model: Optional model for perplexity calculation
        """
        super().__init__()
        self.config = config or QualityConfig()
        self.text_column = text_column
        self.stats.filter_type = FilterType.QUALITY

        # Perplexity model (lazy loaded)
        self._perplexity_model = perplexity_model
        self._perplexity_tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_perplexity_model(self) -> None:
        """Lazy load perplexity model."""
        if self._perplexity_model is None and self.config.max_perplexity is not None:
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers is required for perplexity calculation")

            self._perplexity_model = GPT2LMHeadModel.from_pretrained(
                self.config.perplexity_model_name
            ).to(self._device)
            self._perplexity_model.eval()
            self._perplexity_tokenizer = GPT2TokenizerFast.from_pretrained(
                self.config.perplexity_model_name
            )

    def filter(self, sample: dict[str, Any]) -> tuple[bool, str | None]:
        """Filter a single sample based on quality criteria.

        Args:
            sample: Dictionary containing the sample data

        Returns:
            Tuple of (should_keep, removal_reason)
        """
        text = sample.get(self.text_column, "")

        if not isinstance(text, str):
            self.stats.update(False, "invalid_type")
            return False, "invalid_type"

        # Length check
        if len(text) < self.config.min_length:
            self.stats.update(False, "too_short")
            return False, "too_short"

        if len(text) > self.config.max_length:
            self.stats.update(False, "too_long")
            return False, "too_long"

        # Word count check
        words = text.split()
        if len(words) < self.config.min_words:
            self.stats.update(False, "too_few_words")
            return False, "too_few_words"

        # Punctuation check
        if self.config.require_punctuation:
            if not re.search(r'[.!?]', text):
                self.stats.update(False, "no_punctuation")
                return False, "no_punctuation"

        # Repetition check
        if self.config.remove_repetitive:
            rep_score = self.detect_repetition(text)
            if rep_score > self.config.repetition_threshold:
                self.stats.update(False, "too_repetitive")
                return False, "too_repetitive"

        # Perplexity check
        if self.config.max_perplexity is not None:
            self._load_perplexity_model()
            ppl = self.compute_perplexity(text)
            if ppl > self.config.max_perplexity:
                self.stats.update(False, "high_perplexity")
                return False, "high_perplexity"

        self.stats.update(True)
        return True, None

    def filter_batch(
        self, samples: list[dict[str, Any]]
    ) -> list[tuple[bool, str | None]]:
        """Filter a batch of samples.

        Args:
            samples: List of sample dictionaries

        Returns:
            List of (should_keep, removal_reason) tuples
        """
        results = []

        # For perplexity, batch processing is more efficient
        if self.config.max_perplexity is not None:
            self._load_perplexity_model()

            # First pass: check non-perplexity criteria
            candidates = []
            candidate_indices = []

            for i, sample in enumerate(samples):
                text = sample.get(self.text_column, "")

                if not isinstance(text, str):
                    results.append((False, "invalid_type"))
                    self.stats.update(False, "invalid_type")
                    continue

                # Quick checks first
                passed, reason = self._quick_checks(text)
                if not passed:
                    results.append((False, reason))
                    self.stats.update(False, reason)
                else:
                    candidates.append(text)
                    candidate_indices.append(i)
                    results.append(None)  # Placeholder

            # Batch perplexity computation
            if candidates:
                perplexities = self._batch_perplexity(candidates)

                for idx, ppl in zip(candidate_indices, perplexities):
                    if ppl > self.config.max_perplexity:
                        results[idx] = (False, "high_perplexity")
                        self.stats.update(False, "high_perplexity")
                    else:
                        results[idx] = (True, None)
                        self.stats.update(True)
        else:
            results = [self.filter(sample) for sample in samples]

        return results

    def _quick_checks(self, text: str) -> tuple[bool, str | None]:
        """Run quick quality checks without perplexity."""
        if len(text) < self.config.min_length:
            return False, "too_short"

        if len(text) > self.config.max_length:
            return False, "too_long"

        words = text.split()
        if len(words) < self.config.min_words:
            return False, "too_few_words"

        if self.config.require_punctuation:
            if not re.search(r'[.!?]', text):
                return False, "no_punctuation"

        if self.config.remove_repetitive:
            rep_score = self.detect_repetition(text)
            if rep_score > self.config.repetition_threshold:
                return False, "too_repetitive"

        return True, None

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity score for a text.

        Args:
            text: Input text

        Returns:
            Perplexity score (lower is better)
        """
        if self._perplexity_model is None or self._perplexity_tokenizer is None:
            self._load_perplexity_model()

        encodings = self._perplexity_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._perplexity_model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss

        return torch.exp(loss).item()

    def _batch_perplexity(self, texts: list[str]) -> list[float]:
        """Compute perplexity for a batch of texts."""
        perplexities = []

        for text in texts:
            ppl = self.compute_perplexity(text)
            perplexities.append(ppl)

        return perplexities

    def compute_coherence(self, text: str) -> float:
        """Compute coherence score for a text.

        Uses sentence-level coherence based on consecutive sentence similarity.

        Args:
            text: Input text

        Returns:
            Coherence score (higher is better, 0-1)
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent by default

        # Simple word overlap coherence
        total_coherence = 0.0
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i + 1].lower().split())

            if not words1 or not words2:
                continue

            overlap = len(words1 & words2) / min(len(words1), len(words2))
            total_coherence += overlap

        return total_coherence / (len(sentences) - 1)

    def detect_repetition(self, text: str, n: int = 3) -> float:
        """Detect repetitive patterns in text using n-gram analysis.

        Args:
            text: Input text
            n: N-gram size for repetition detection

        Returns:
            Repetition score (0-1, higher means more repetitive)
        """
        words = text.lower().split()

        if len(words) < n:
            return 0.0

        # Generate n-grams
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

        if not ngrams:
            return 0.0

        # Count n-gram frequencies
        ngram_counts = Counter(ngrams)

        # Calculate repetition ratio
        total_ngrams = len(ngrams)
        unique_ngrams = len(ngram_counts)

        if total_ngrams == 0:
            return 0.0

        # Repetition score: 1 - (unique / total)
        return 1.0 - (unique_ngrams / total_ngrams)


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication.

    Attributes:
        method: Deduplication method ("minhash", "exact")
        threshold: Similarity threshold for near-duplicate detection (Jaccard)
        num_perm: Number of permutations for MinHash
        ngram_size: N-gram size for shingling
        seed: Random seed for reproducibility
    """

    method: str = "minhash"
    threshold: float = 0.8
    num_perm: int = 128
    ngram_size: int = 5
    seed: int = 42


class Deduplicator(BaseFilter):
    """Near-duplicate detection and removal using MinHash LSH.

    This filter identifies and removes near-duplicate documents to improve
    training data diversity. Uses Locality Sensitive Hashing for efficient
    similarity search at scale.

    Args:
        config: DeduplicationConfig with parameters
        text_column: Name of the column containing text to deduplicate
    """

    def __init__(
        self,
        config: DeduplicationConfig | None = None,
        text_column: str = "text",
    ) -> None:
        """Initialize the deduplicator.

        Args:
            config: Deduplication configuration
            text_column: Column name containing text to deduplicate
        """
        super().__init__()
        self.config = config or DeduplicationConfig()
        self.text_column = text_column
        self.stats.filter_type = FilterType.DEDUP

        # For exact deduplication
        self._seen_hashes: set[str] = set()

        # For MinHash LSH
        self._lsh: MinHashLSH | None = None
        self._minhash_cache: dict[int, MinHash] = {}
        self._doc_id = 0

        if self.config.method == "minhash":
            if not _DATASKETCH_AVAILABLE:
                raise ImportError("datasketch is required for MinHash deduplication")
            self._lsh = MinHashLSH(
                threshold=self.config.threshold,
                num_perm=self.config.num_perm,
            )

    def filter(self, sample: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if sample is a duplicate of previously seen samples.

        Args:
            sample: Dictionary containing the sample data

        Returns:
            Tuple of (should_keep, removal_reason)
        """
        text = sample.get(self.text_column, "")

        if not isinstance(text, str):
            self.stats.update(False, "invalid_type")
            return False, "invalid_type"

        if self.config.method == "exact":
            return self._filter_exact(text)
        elif self.config.method == "minhash":
            return self._filter_minhash(text)
        else:
            raise ValueError(f"Unknown deduplication method: {self.config.method}")

    def _filter_exact(self, text: str) -> tuple[bool, str | None]:
        """Exact deduplication using hash."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        if text_hash in self._seen_hashes:
            self.stats.update(False, "exact_duplicate")
            return False, "exact_duplicate"

        self._seen_hashes.add(text_hash)
        self.stats.update(True)
        return True, None

    def _filter_minhash(self, text: str) -> tuple[bool, str | None]:
        """Near-duplicate detection using MinHash LSH."""
        minhash = self.compute_minhash(text)

        # Query for similar documents
        similar = self._lsh.query(minhash)

        if similar:
            self.stats.update(False, "near_duplicate")
            return False, "near_duplicate"

        # Add to index
        self._lsh.insert(str(self._doc_id), minhash)
        self._minhash_cache[self._doc_id] = minhash
        self._doc_id += 1

        self.stats.update(True)
        return True, None

    def filter_batch(
        self, samples: list[dict[str, Any]]
    ) -> list[tuple[bool, str | None]]:
        """Filter duplicates from a batch of samples.

        Args:
            samples: List of sample dictionaries

        Returns:
            List of (should_keep, removal_reason) tuples
        """
        return [self.filter(sample) for sample in samples]

    def compute_minhash(self, text: str) -> "MinHash":
        """Compute MinHash signature for a text.

        Args:
            text: Input text

        Returns:
            MinHash signature object
        """
        if not _DATASKETCH_AVAILABLE:
            raise ImportError("datasketch is required for MinHash computation")

        minhash = MinHash(num_perm=self.config.num_perm, seed=self.config.seed)

        # Generate n-gram shingles
        words = text.lower().split()
        for i in range(max(1, len(words) - self.config.ngram_size + 1)):
            shingle = " ".join(words[i:i + self.config.ngram_size])
            minhash.update(shingle.encode('utf-8'))

        return minhash

    def build_lsh_index(self, texts: list[str]) -> None:
        """Build LSH index from a list of texts.

        Args:
            texts: List of texts to index
        """
        if not _DATASKETCH_AVAILABLE:
            raise ImportError("datasketch is required for LSH indexing")

        self.reset_index()

        for text in tqdm(texts, desc="Building LSH index"):
            minhash = self.compute_minhash(text)
            self._lsh.insert(str(self._doc_id), minhash)
            self._minhash_cache[self._doc_id] = minhash
            self._doc_id += 1

    def query_similar(self, text: str) -> list[str]:
        """Query LSH index for similar documents.

        Args:
            text: Text to query

        Returns:
            List of document IDs of similar documents
        """
        minhash = self.compute_minhash(text)
        return self._lsh.query(minhash)

    def reset_index(self) -> None:
        """Reset the LSH index and hash cache."""
        self._seen_hashes = set()
        self._minhash_cache = {}
        self._doc_id = 0

        if self.config.method == "minhash":
            self._lsh = MinHashLSH(
                threshold=self.config.threshold,
                num_perm=self.config.num_perm,
            )


@dataclass
class ToxicityConfig:
    """Configuration for toxicity filtering.

    Attributes:
        model_name: Name of detoxify model ("original", "unbiased", "multilingual")
        threshold: Toxicity score threshold for removal (default 0.7)
        categories: Specific toxicity categories to filter (if None, use all)
        device: Device for model inference
    """

    model_name: str = "original"
    threshold: float = 0.7
    categories: list[str] | None = None
    device: str | None = None  # Auto-detect


class ToxicityFilter(BaseFilter):
    """Filter for removing toxic, harmful, or unsafe content.

    Uses detoxify library to detect and remove content that
    is toxic, hateful, sexually explicit, or otherwise harmful.

    Args:
        config: ToxicityConfig with filter parameters
        text_column: Name of the column containing text to filter
    """

    # Default categories from detoxify
    ALL_CATEGORIES = [
        "toxicity",
        "severe_toxicity",
        "obscene",
        "threat",
        "insult",
        "identity_attack",
        "sexual_explicit",
    ]

    def __init__(
        self,
        config: ToxicityConfig | None = None,
        text_column: str = "text",
    ) -> None:
        """Initialize the toxicity filter.

        Args:
            config: Toxicity filter configuration
            text_column: Column name containing text to filter
        """
        super().__init__()

        if not _DETOXIFY_AVAILABLE:
            raise ImportError(
                "detoxify is required for toxicity filtering. "
                "Install with: pip install detoxify"
            )

        self.config = config or ToxicityConfig()
        self.text_column = text_column
        self.stats.filter_type = FilterType.TOXICITY

        # Categories to check
        self.categories = self.config.categories or self.ALL_CATEGORIES

        # Load model
        device = self.config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = Detoxify(self.config.model_name, device=device)

    def filter(self, sample: dict[str, Any]) -> tuple[bool, str | None]:
        """Filter a single sample based on toxicity.

        Args:
            sample: Dictionary containing the sample data

        Returns:
            Tuple of (should_keep, removal_reason)
        """
        text = sample.get(self.text_column, "")

        if not isinstance(text, str):
            self.stats.update(False, "invalid_type")
            return False, "invalid_type"

        scores = self._model.predict(text)

        for category in self.categories:
            if category in scores:
                if scores[category] > self.config.threshold:
                    reason = f"toxic_{category}"
                    self.stats.update(False, reason)
                    return False, reason

        self.stats.update(True)
        return True, None

    def filter_batch(
        self, samples: list[dict[str, Any]]
    ) -> list[tuple[bool, str | None]]:
        """Filter a batch of samples for toxicity.

        Args:
            samples: List of sample dictionaries

        Returns:
            List of (should_keep, removal_reason) tuples
        """
        texts = []
        valid_indices = []
        results: list[tuple[bool, str | None]] = [(False, "invalid_type")] * len(samples)

        for i, sample in enumerate(samples):
            text = sample.get(self.text_column, "")
            if isinstance(text, str) and text.strip():
                texts.append(text)
                valid_indices.append(i)

        if not texts:
            for i in range(len(samples)):
                self.stats.update(False, "invalid_type")
            return results

        # Batch prediction
        all_scores = self.compute_toxicity_scores(texts)

        for idx, scores in zip(valid_indices, all_scores):
            passed = True
            reason = None

            for category in self.categories:
                if category in scores:
                    if scores[category] > self.config.threshold:
                        passed = False
                        reason = f"toxic_{category}"
                        break

            results[idx] = (passed, reason)
            self.stats.update(passed, reason)

        # Update stats for invalid samples
        for i in range(len(samples)):
            if i not in valid_indices:
                self.stats.update(False, "invalid_type")

        return results

    def compute_toxicity_scores(self, texts: list[str]) -> list[dict[str, float]]:
        """Compute toxicity scores for a list of texts.

        Args:
            texts: List of input texts

        Returns:
            List of dictionaries mapping toxicity categories to scores
        """
        results = self._model.predict(texts)

        # Convert from {category: [scores]} to [{category: score}]
        num_texts = len(texts)
        scores_list = []

        for i in range(num_texts):
            scores = {}
            for category, values in results.items():
                if isinstance(values, list):
                    scores[category] = values[i]
                else:
                    scores[category] = values
            scores_list.append(scores)

        return scores_list


class DataCurator:
    """Main class for orchestrating data curation pipeline.

    Combines multiple filters (quality, deduplication, toxicity) into a
    unified curation pipeline. Supports streaming and batch processing.

    Args:
        filters: List of filters to apply in order
        text_column: Name of the column containing text to process
        num_workers: Number of parallel workers for processing
    """

    def __init__(
        self,
        filters: list[BaseFilter] | None = None,
        text_column: str = "text",
        num_workers: int = 4,
    ) -> None:
        """Initialize the data curator.

        Args:
            filters: List of filters to apply
            text_column: Column name containing text
            num_workers: Number of parallel workers
        """
        self.filters = filters or []
        self.text_column = text_column
        self.num_workers = num_workers
        self._filter_stats: dict[str, FilterStats] = {}

    def add_filter(self, filter_: BaseFilter) -> "DataCurator":
        """Add a filter to the pipeline.

        Args:
            filter_: Filter to add

        Returns:
            Self for method chaining
        """
        self.filters.append(filter_)
        return self

    def process_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 1000,
        show_progress: bool = True,
    ) -> tuple[Dataset, dict[str, FilterStats]]:
        """Process a HuggingFace dataset through the curation pipeline.

        Args:
            dataset: Input dataset
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (filtered_dataset, filter_statistics)
        """
        if not self.filters:
            return dataset, {}

        # Reset stats
        self.reset_stats()

        # Keep track of which indices to keep
        keep_indices: list[int] = []

        # Process in batches
        total_samples = len(dataset)

        iterator = range(0, total_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Curating data")

        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, total_samples)
            batch = dataset.select(range(start_idx, end_idx))

            # Convert to list of dicts for processing
            samples = [batch[i] for i in range(len(batch))]

            # Apply each filter in sequence
            current_indices = list(range(len(samples)))

            for filter_ in self.filters:
                if not current_indices:
                    break

                # Get samples that are still candidates
                candidate_samples = [samples[i] for i in current_indices]
                results = filter_.filter_batch(candidate_samples)

                # Update indices to keep
                new_indices = []
                for i, (passed, _) in zip(current_indices, results):
                    if passed:
                        new_indices.append(i)

                current_indices = new_indices

            # Add global indices for samples that passed all filters
            keep_indices.extend([start_idx + i for i in current_indices])

        # Create filtered dataset
        filtered_dataset = dataset.select(keep_indices)

        # Collect stats
        stats = self.get_stats()

        return filtered_dataset, stats

    def process_streaming(
        self,
        data_iterator: Iterator[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        """Process streaming data through the curation pipeline.

        Args:
            data_iterator: Iterator yielding sample dictionaries

        Yields:
            Samples that pass all filters
        """
        for sample in data_iterator:
            passed = True

            for filter_ in self.filters:
                result, _ = filter_.filter(sample)
                if not result:
                    passed = False
                    break

            if passed:
                yield sample

    def get_stats(self) -> dict[str, FilterStats]:
        """Get statistics from all filters.

        Returns:
            Dictionary mapping filter names to their statistics
        """
        stats = {}
        for i, filter_ in enumerate(self.filters):
            filter_name = filter_.__class__.__name__
            if filter_name in stats:
                filter_name = f"{filter_name}_{i}"
            stats[filter_name] = filter_.stats
        return stats

    def reset_stats(self) -> None:
        """Reset statistics for all filters."""
        for filter_ in self.filters:
            filter_.reset_stats()

    @classmethod
    def from_config(
        cls,
        quality_config: QualityConfig | None = None,
        dedup_config: DeduplicationConfig | None = None,
        toxicity_config: ToxicityConfig | None = None,
        text_column: str = "text",
        **kwargs: Any,
    ) -> "DataCurator":
        """Create a DataCurator from configuration objects.

        Args:
            quality_config: Quality filter configuration
            dedup_config: Deduplication configuration
            toxicity_config: Toxicity filter configuration
            text_column: Column name containing text
            **kwargs: Additional arguments for DataCurator

        Returns:
            Configured DataCurator instance
        """
        filters: list[BaseFilter] = []

        if quality_config is not None:
            filters.append(QualityFilter(config=quality_config, text_column=text_column))

        if dedup_config is not None:
            filters.append(Deduplicator(config=dedup_config, text_column=text_column))

        if toxicity_config is not None:
            filters.append(ToxicityFilter(config=toxicity_config, text_column=text_column))

        return cls(filters=filters, text_column=text_column, **kwargs)


def filter_quality(
    dataset: Dataset,
    min_length: int = 50,
    max_length: int = 100000,
    min_words: int = 10,
    max_perplexity: float | None = None,
    remove_repetitive: bool = True,
    repetition_threshold: float = 0.3,
    text_column: str = "text",
) -> tuple[Dataset, FilterStats]:
    """Convenience function for quality filtering.

    Args:
        dataset: Input HuggingFace dataset
        min_length: Minimum text length
        max_length: Maximum text length
        min_words: Minimum word count
        max_perplexity: Maximum perplexity (None to skip)
        remove_repetitive: Whether to filter repetitive texts
        repetition_threshold: Threshold for repetition detection
        text_column: Column containing text

    Returns:
        Tuple of (filtered_dataset, statistics)
    """
    config = QualityConfig(
        min_length=min_length,
        max_length=max_length,
        min_words=min_words,
        max_perplexity=max_perplexity,
        remove_repetitive=remove_repetitive,
        repetition_threshold=repetition_threshold,
    )

    curator = DataCurator.from_config(quality_config=config, text_column=text_column)
    return curator.process_dataset(dataset)


def deduplicate(
    dataset: Dataset,
    method: str = "minhash",
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram_size: int = 5,
    text_column: str = "text",
) -> tuple[Dataset, FilterStats]:
    """Convenience function for deduplication.

    Args:
        dataset: Input HuggingFace dataset
        method: Deduplication method ("minhash" or "exact")
        threshold: Jaccard similarity threshold for near-duplicates
        num_perm: Number of MinHash permutations
        ngram_size: N-gram size for shingling
        text_column: Column containing text

    Returns:
        Tuple of (deduplicated_dataset, statistics)
    """
    config = DeduplicationConfig(
        method=method,
        threshold=threshold,
        num_perm=num_perm,
        ngram_size=ngram_size,
    )

    curator = DataCurator.from_config(dedup_config=config, text_column=text_column)
    return curator.process_dataset(dataset)


def remove_toxic(
    dataset: Dataset,
    threshold: float = 0.7,
    categories: list[str] | None = None,
    text_column: str = "text",
) -> tuple[Dataset, FilterStats]:
    """Convenience function for toxicity filtering.

    Args:
        dataset: Input HuggingFace dataset
        threshold: Toxicity score threshold for removal
        categories: Specific categories to filter (None for all)
        text_column: Column containing text

    Returns:
        Tuple of (filtered_dataset, statistics)
    """
    config = ToxicityConfig(
        threshold=threshold,
        categories=categories,
    )

    curator = DataCurator.from_config(toxicity_config=config, text_column=text_column)
    return curator.process_dataset(dataset)
