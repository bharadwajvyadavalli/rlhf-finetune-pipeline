"""
Preference pair generation for reward modeling and DPO.

This module provides tools for creating preference datasets with
chosen/rejected pairs from various sources:
- Model-generated comparisons with quality ranking
- Human preference data integration
- Constitutional AI-style self-critique

These preference pairs are used to train reward models and for
Direct Preference Optimization (DPO).
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

# Lazy imports
_TRANSFORMERS_AVAILABLE = False
_OPENAI_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

try:
    import openai
    _OPENAI_AVAILABLE = True
except ImportError:
    pass


class PairGenerationStrategy(Enum):
    """Strategies for generating preference pairs."""

    MODEL_RANKING = "model_ranking"
    REWARD_MODEL = "reward_model"
    SELF_CRITIQUE = "self_critique"
    HUMAN_LABELS = "human_labels"
    CONSTITUTIONAL = "constitutional"


@dataclass
class PreferencePairConfig:
    """Configuration for preference pair generation.

    Attributes:
        strategy: Strategy for generating pairs
        num_responses: Number of responses to generate per prompt
        model_name: Model for response generation
        reward_model_name: Reward model for ranking (if using reward strategy)
        temperatures: List of temperatures for diverse response generation
        max_new_tokens: Maximum tokens to generate
        constitutional_principles: Principles for constitutional AI approach
        ranking_model_name: Model for ranking responses
        use_openai: Whether to use OpenAI API
        openai_model: OpenAI model to use
        margin: Minimum score margin between chosen and rejected
    """

    strategy: PairGenerationStrategy = PairGenerationStrategy.MODEL_RANKING
    num_responses: int = 4
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    reward_model_name: str | None = None
    temperatures: list[float] = field(default_factory=lambda: [0.3, 0.7, 1.0])
    max_new_tokens: int = 512
    constitutional_principles: list[str] | None = None
    ranking_model_name: str | None = None
    use_openai: bool = False
    openai_model: str = "gpt-3.5-turbo"
    margin: float = 1.0


@dataclass
class PreferencePair:
    """A single preference pair.

    Attributes:
        prompt: The input prompt
        chosen: The preferred response
        rejected: The rejected response
        chosen_score: Optional score for chosen response
        rejected_score: Optional score for rejected response
        metadata: Optional additional metadata
    """

    prompt: str
    chosen: str
    rejected: str
    chosen_score: float | None = None
    rejected_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            **self.metadata,
        }


class BasePreferenceGenerator(ABC):
    """Abstract base class for preference pair generators."""

    def __init__(self, config: PreferencePairConfig | None = None) -> None:
        self.config = config or PreferencePairConfig()
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self._openai_client = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def generate_pairs(
        self, prompts: list[str], num_pairs_per_prompt: int = 1
    ) -> list[PreferencePair]:
        """Generate preference pairs for a list of prompts.

        Args:
            prompts: List of input prompts
            num_pairs_per_prompt: Number of pairs to generate per prompt

        Returns:
            List of PreferencePair objects
        """
        pass

    @abstractmethod
    def rank_responses(
        self, prompt: str, responses: list[str]
    ) -> list[tuple[str, float]]:
        """Rank a list of responses for a prompt.

        Args:
            prompt: The input prompt
            responses: List of candidate responses

        Returns:
            List of (response, score) tuples sorted by score descending
        """
        pass

    def load_model(self) -> None:
        """Load the generation model."""
        if self.config.use_openai:
            if not _OPENAI_AVAILABLE:
                raise ImportError("openai package is required")
            self._openai_client = openai.OpenAI()
        else:
            if not _TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers is required")

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None,
            )

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )

    def generate_response(
        self, prompt: str, temperature: float = 0.7
    ) -> str:
        """Generate a single response.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature

        Returns:
            Generated response
        """
        if self.config.use_openai:
            if self._openai_client is None:
                self.load_model()

            response = self._openai_client.chat.completions.create(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=self.config.max_new_tokens,
            )
            return response.choices[0].message.content
        else:
            if self._pipeline is None:
                self.load_model()

            output = self._pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

            generated = output[0]["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            return generated


class ModelRankingGenerator(BasePreferenceGenerator):
    """Generate preference pairs by having a model rank multiple responses.

    Generates multiple responses for each prompt, then uses a ranking
    model to order them by quality. Pairs are created from the ranked list.
    """

    RANKING_PROMPT = """You are a helpful assistant that evaluates response quality.

Given the prompt and two responses, rate each response from 1-10 based on:
- Helpfulness and relevance
- Accuracy and correctness
- Clarity and coherence
- Completeness

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Output your ratings in this exact format:
Response A: [score]
Response B: [score]

Ratings:"""

    def __init__(
        self,
        config: PreferencePairConfig | None = None,
        generation_model: Any | None = None,
        ranking_model: Any | None = None,
    ) -> None:
        """Initialize the model ranking generator.

        Args:
            config: Generation configuration
            generation_model: Response generation model
            ranking_model: Response ranking model
        """
        super().__init__(config)
        self._generation_model = generation_model
        self._ranking_model = ranking_model
        self._ranking_pipeline = None

    def generate_pairs(
        self, prompts: list[str], num_pairs_per_prompt: int = 1
    ) -> list[PreferencePair]:
        """Generate preference pairs using model ranking.

        Args:
            prompts: List of input prompts
            num_pairs_per_prompt: Number of pairs per prompt

        Returns:
            List of preference pairs
        """
        pairs = []

        for prompt in tqdm(prompts, desc="Generating preference pairs"):
            # Generate multiple responses at different temperatures
            responses = self.generate_responses(prompt, self.config.num_responses)

            if len(responses) < 2:
                continue

            # Rank responses
            ranked = self.rank_responses(prompt, responses)

            # Create pairs from ranked responses
            for i in range(min(num_pairs_per_prompt, len(ranked) - 1)):
                chosen_resp, chosen_score = ranked[i]
                rejected_resp, rejected_score = ranked[-(i + 1)]

                pair = PreferencePair(
                    prompt=prompt,
                    chosen=chosen_resp,
                    rejected=rejected_resp,
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                )
                pairs.append(pair)

        return pairs

    def rank_responses(
        self, prompt: str, responses: list[str]
    ) -> list[tuple[str, float]]:
        """Rank responses using pairwise comparison.

        Args:
            prompt: The input prompt
            responses: List of candidate responses

        Returns:
            List of (response, score) tuples sorted descending
        """
        if len(responses) < 2:
            return [(r, 1.0) for r in responses]

        # Score each response using pairwise comparisons
        scores = {r: 0.0 for r in responses}

        for i, resp_a in enumerate(responses):
            for j, resp_b in enumerate(responses):
                if i >= j:
                    continue

                # Compare pair
                score_a, score_b = self._compare_responses(prompt, resp_a, resp_b)
                scores[resp_a] += score_a
                scores[resp_b] += score_b

        # Normalize and sort
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {r: s / max_score for r, s in scores.items()}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def _compare_responses(
        self, prompt: str, response_a: str, response_b: str
    ) -> tuple[float, float]:
        """Compare two responses and return scores."""
        ranking_prompt = self.RANKING_PROMPT.format(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
        )

        try:
            result = self.generate_response(ranking_prompt, temperature=0.1)

            # Parse scores
            import re
            score_a_match = re.search(r'Response A:\s*(\d+(?:\.\d+)?)', result)
            score_b_match = re.search(r'Response B:\s*(\d+(?:\.\d+)?)', result)

            score_a = float(score_a_match.group(1)) if score_a_match else 5.0
            score_b = float(score_b_match.group(1)) if score_b_match else 5.0

            return score_a, score_b

        except Exception:
            # Default to equal scores on failure
            return 5.0, 5.0

    def generate_responses(
        self, prompt: str, num_responses: int
    ) -> list[str]:
        """Generate multiple responses at different temperatures.

        Args:
            prompt: The input prompt
            num_responses: Number of responses to generate

        Returns:
            List of generated responses
        """
        responses = []
        temperatures = self.config.temperatures

        for i in range(num_responses):
            temp = temperatures[i % len(temperatures)]
            try:
                response = self.generate_response(prompt, temperature=temp)
                if response and len(response.strip()) > 10:
                    responses.append(response)
            except Exception:
                continue

        return responses


class RewardModelGenerator(BasePreferenceGenerator):
    """Generate preference pairs using a trained reward model.

    Uses a reward model to score responses and create preference pairs
    based on reward scores.
    """

    def __init__(
        self,
        config: PreferencePairConfig | None = None,
        reward_model: Any | None = None,
        generation_model: Any | None = None,
    ) -> None:
        """Initialize the reward model generator.

        Args:
            config: Generation configuration
            reward_model: Trained reward model
            generation_model: Response generation model
        """
        super().__init__(config)
        self._reward_model = reward_model
        self._reward_tokenizer = None

    def generate_pairs(
        self, prompts: list[str], num_pairs_per_prompt: int = 1
    ) -> list[PreferencePair]:
        """Generate preference pairs using reward model scores.

        Args:
            prompts: List of input prompts
            num_pairs_per_prompt: Number of pairs per prompt

        Returns:
            List of preference pairs
        """
        pairs = []

        for prompt in tqdm(prompts, desc="Generating preference pairs"):
            # Generate multiple responses
            responses = []
            for temp in self.config.temperatures:
                for _ in range(self.config.num_responses // len(self.config.temperatures)):
                    resp = self.generate_response(prompt, temperature=temp)
                    if resp and len(resp.strip()) > 10:
                        responses.append(resp)

            if len(responses) < 2:
                continue

            # Rank using reward model
            ranked = self.rank_responses(prompt, responses)

            # Create pairs
            for i in range(min(num_pairs_per_prompt, len(ranked) - 1)):
                chosen_resp, chosen_score = ranked[i]
                rejected_resp, rejected_score = ranked[-(i + 1)]

                # Apply margin filter
                if chosen_score - rejected_score < self.config.margin:
                    continue

                pair = PreferencePair(
                    prompt=prompt,
                    chosen=chosen_resp,
                    rejected=rejected_resp,
                    chosen_score=chosen_score,
                    rejected_score=rejected_score,
                )
                pairs.append(pair)

        return pairs

    def rank_responses(
        self, prompt: str, responses: list[str]
    ) -> list[tuple[str, float]]:
        """Rank responses using reward model scores.

        Args:
            prompt: The input prompt
            responses: List of candidate responses

        Returns:
            List of (response, score) tuples sorted descending
        """
        if self._reward_model is None:
            raise ValueError("Reward model not loaded")

        prompts = [prompt] * len(responses)
        scores = self.compute_rewards(prompts, responses)

        ranked = sorted(zip(responses, scores), key=lambda x: x[1], reverse=True)
        return ranked

    def compute_rewards(
        self, prompts: list[str], responses: list[str]
    ) -> list[float]:
        """Compute reward scores for prompt-response pairs.

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            List of reward scores
        """
        if self._reward_model is None:
            raise ValueError("Reward model not loaded")

        # Combine prompts and responses
        texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]

        # Tokenize
        if self._reward_tokenizer is None:
            self._reward_tokenizer = AutoTokenizer.from_pretrained(
                self.config.reward_model_name or self.config.model_name
            )

        encodings = self._reward_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self._device)

        # Get rewards
        with torch.no_grad():
            outputs = self._reward_model(**encodings)
            # Assume reward model outputs logits/rewards
            if hasattr(outputs, 'logits'):
                rewards = outputs.logits.squeeze(-1)
            else:
                rewards = outputs[0].squeeze(-1)

        return rewards.cpu().tolist()


class ConstitutionalGenerator(BasePreferenceGenerator):
    """Generate preference pairs using Constitutional AI self-critique.

    Uses the model to critique and revise its own responses based on
    a set of constitutional principles.
    """

    DEFAULT_PRINCIPLES = [
        "Please choose the response that is the most helpful, harmless, and honest.",
        "Please choose the response that is less harmful or toxic.",
        "Please choose the response that is more accurate and factual.",
        "Please choose the response that is more respectful and considerate.",
    ]

    CRITIQUE_PROMPT = """You are a helpful assistant that critiques responses.

Given this response to a prompt, critique it based on the following principle:
Principle: {principle}

Prompt: {prompt}

Response: {response}

Provide a brief critique identifying any issues with the response based on the principle.
Critique:"""

    REVISION_PROMPT = """You are a helpful assistant that improves responses.

Given the original response and critique, provide an improved version.

Prompt: {prompt}

Original Response: {response}

Critique: {critique}

Improved Response:"""

    def __init__(
        self,
        config: PreferencePairConfig | None = None,
        model: Any | None = None,
        principles: list[str] | None = None,
    ) -> None:
        """Initialize the constitutional generator.

        Args:
            config: Generation configuration
            model: Generation/critique model
            principles: Constitutional principles
        """
        super().__init__(config)
        self.principles = principles or self.config.constitutional_principles or self.DEFAULT_PRINCIPLES

    def generate_pairs(
        self, prompts: list[str], num_pairs_per_prompt: int = 1
    ) -> list[PreferencePair]:
        """Generate preference pairs using constitutional AI.

        Args:
            prompts: List of input prompts
            num_pairs_per_prompt: Number of pairs per prompt

        Returns:
            List of preference pairs
        """
        pairs = []

        for prompt in tqdm(prompts, desc="Generating constitutional pairs"):
            for _ in range(num_pairs_per_prompt):
                # Generate initial response
                initial_response = self.generate_response(prompt, temperature=0.7)

                if not initial_response:
                    continue

                # Select a random principle
                principle = random.choice(self.principles)

                # Critique the response
                critique = self.critique_response(prompt, initial_response, principle)

                # Revise based on critique
                revised_response = self.revise_response(prompt, initial_response, critique)

                if not revised_response:
                    continue

                # Revised is chosen, original is rejected
                pair = PreferencePair(
                    prompt=prompt,
                    chosen=revised_response,
                    rejected=initial_response,
                    metadata={"principle": principle, "critique": critique},
                )
                pairs.append(pair)

        return pairs

    def rank_responses(
        self, prompt: str, responses: list[str]
    ) -> list[tuple[str, float]]:
        """Rank responses based on constitutional principles.

        Args:
            prompt: The input prompt
            responses: List of candidate responses

        Returns:
            List of (response, score) tuples sorted descending
        """
        # Score each response based on principles
        scores = []

        for response in responses:
            score = 0.0
            for principle in self.principles:
                critique = self.critique_response(prompt, response, principle)
                # Simple heuristic: shorter critique = fewer issues = better
                score += 1.0 / (1.0 + len(critique) / 100.0)
            scores.append((response, score / len(self.principles)))

        return sorted(scores, key=lambda x: x[1], reverse=True)

    def critique_response(
        self, prompt: str, response: str, principle: str
    ) -> str:
        """Critique a response based on a principle.

        Args:
            prompt: The input prompt
            response: Response to critique
            principle: Constitutional principle

        Returns:
            Critique text
        """
        critique_prompt = self.CRITIQUE_PROMPT.format(
            principle=principle,
            prompt=prompt,
            response=response,
        )

        return self.generate_response(critique_prompt, temperature=0.3)

    def revise_response(
        self, prompt: str, response: str, critique: str
    ) -> str:
        """Revise a response based on critique.

        Args:
            prompt: The input prompt
            response: Response to revise
            critique: Critique text

        Returns:
            Revised response
        """
        revision_prompt = self.REVISION_PROMPT.format(
            prompt=prompt,
            response=response,
            critique=critique,
        )

        return self.generate_response(revision_prompt, temperature=0.5)


class PreferencePairGenerator:
    """Unified interface for preference pair generation.

    Combines multiple strategies for generating preference pairs.
    """

    def __init__(
        self,
        config: PreferencePairConfig | None = None,
        strategy: PairGenerationStrategy | None = None,
    ) -> None:
        """Initialize the preference pair generator.

        Args:
            config: Generation configuration
            strategy: Generation strategy
        """
        self.config = config or PreferencePairConfig()
        if strategy:
            self.config.strategy = strategy

        self._generator: BasePreferenceGenerator | None = None

    def _get_generator(self) -> BasePreferenceGenerator:
        """Get the appropriate generator for the strategy."""
        if self._generator is not None:
            return self._generator

        if self.config.strategy == PairGenerationStrategy.MODEL_RANKING:
            self._generator = ModelRankingGenerator(self.config)
        elif self.config.strategy == PairGenerationStrategy.REWARD_MODEL:
            self._generator = RewardModelGenerator(self.config)
        elif self.config.strategy == PairGenerationStrategy.CONSTITUTIONAL:
            self._generator = ConstitutionalGenerator(self.config)
        else:
            self._generator = ModelRankingGenerator(self.config)

        return self._generator

    def generate(
        self,
        prompts: list[str],
        num_pairs_per_prompt: int = 1,
    ) -> Dataset:
        """Generate preference pairs for prompts.

        Args:
            prompts: List of input prompts
            num_pairs_per_prompt: Pairs to generate per prompt

        Returns:
            Dataset with preference pairs
        """
        generator = self._get_generator()
        pairs = generator.generate_pairs(prompts, num_pairs_per_prompt)

        # Convert to dataset
        data = [pair.to_dict() for pair in pairs]
        return Dataset.from_list(data)

    def from_human_labels(
        self,
        labeled_data: list[dict[str, Any]],
    ) -> Dataset:
        """Create preference dataset from human-labeled data.

        Expected format:
        [
            {"prompt": "...", "chosen": "...", "rejected": "..."},
            ...
        ]

        Args:
            labeled_data: List of labeled preference dictionaries

        Returns:
            Dataset with preference pairs
        """
        return Dataset.from_list(labeled_data)

    def from_ratings(
        self,
        rated_data: list[dict[str, Any]],
        rating_column: str = "rating",
        margin: float = 1.0,
    ) -> Dataset:
        """Convert Likert-scale ratings to pairwise preferences.

        Args:
            rated_data: List of dicts with prompt, response, rating
            rating_column: Name of the rating column
            margin: Minimum rating difference for a valid pair

        Returns:
            Dataset with preference pairs
        """
        # Group by prompt
        from collections import defaultdict
        grouped = defaultdict(list)

        for item in rated_data:
            grouped[item["prompt"]].append(item)

        # Create pairs from ratings
        pairs = []
        for prompt, items in grouped.items():
            # Sort by rating descending
            items = sorted(items, key=lambda x: x[rating_column], reverse=True)

            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if items[i][rating_column] - items[j][rating_column] >= margin:
                        pairs.append({
                            "prompt": prompt,
                            "chosen": items[i].get("response", items[i].get("text", "")),
                            "rejected": items[j].get("response", items[j].get("text", "")),
                            "chosen_score": items[i][rating_column],
                            "rejected_score": items[j][rating_column],
                        })

        return Dataset.from_list(pairs)

    def save_dataset(
        self,
        dataset: Dataset,
        output_path: str,
        format: str = "json",
    ) -> None:
        """Save preference dataset to disk.

        Args:
            dataset: Dataset to save
            output_path: Output file path
            format: Output format
        """
        if format == "json":
            dataset.to_json(output_path)
        elif format == "parquet":
            dataset.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")


class PreferenceDataset(TorchDataset):
    """PyTorch Dataset for preference data.

    Handles tokenization and formatting of preference pairs for
    reward model training or DPO.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Any,
        max_length: int = 2048,
        chosen_column: str = "chosen",
        rejected_column: str = "rejected",
        prompt_column: str | None = "prompt",
    ) -> None:
        """Initialize the preference dataset.

        Args:
            dataset: HuggingFace dataset
            tokenizer: Tokenizer for encoding
            max_length: Max sequence length
            chosen_column: Chosen response column
            rejected_column: Rejected response column
            prompt_column: Prompt column
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chosen_column = chosen_column
        self.rejected_column = rejected_column
        self.prompt_column = prompt_column

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single tokenized sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with tokenized chosen and rejected inputs
        """
        item = self.dataset[idx]

        prompt = item.get(self.prompt_column, "") if self.prompt_column else ""
        chosen = item[self.chosen_column]
        rejected = item[self.rejected_column]

        return self.tokenize_pair(prompt, chosen, rejected)

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
            Dictionary with tokenized inputs
        """
        # Combine prompt and responses
        chosen_text = f"{prompt}\n{chosen}" if prompt else chosen
        rejected_text = f"{prompt}\n{rejected}" if prompt else rejected

        # Tokenize chosen
        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize rejected
        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }

    def collate_fn(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary
        """
        return {
            "chosen_input_ids": torch.stack([b["chosen_input_ids"] for b in batch]),
            "chosen_attention_mask": torch.stack([b["chosen_attention_mask"] for b in batch]),
            "rejected_input_ids": torch.stack([b["rejected_input_ids"] for b in batch]),
            "rejected_attention_mask": torch.stack([b["rejected_attention_mask"] for b in batch]),
        }


def generate_pairs(
    prompts: list[str],
    model_name: str = "gpt-3.5-turbo",
    n_candidates: int = 4,
    strategy: str = "model_ranking",
    use_openai: bool = True,
) -> Dataset:
    """Convenience function to generate preference pairs.

    Args:
        prompts: List of prompts
        model_name: Model to use
        n_candidates: Number of candidate responses per prompt
        strategy: Generation strategy
        use_openai: Whether to use OpenAI API

    Returns:
        Dataset with preference pairs
    """
    config = PreferencePairConfig(
        strategy=PairGenerationStrategy(strategy),
        num_responses=n_candidates,
        model_name=model_name,
        use_openai=use_openai,
        openai_model=model_name if use_openai else "gpt-3.5-turbo",
    )

    generator = PreferencePairGenerator(config)
    return generator.generate(prompts)


def from_ratings(
    rated_data: list[dict[str, Any]],
    margin: float = 1.0,
) -> Dataset:
    """Convenience function to convert ratings to preferences.

    Args:
        rated_data: List of rated items
        margin: Minimum rating margin

    Returns:
        Dataset with preference pairs
    """
    generator = PreferencePairGenerator()
    return generator.from_ratings(rated_data, margin=margin)
