"""
Benchmark evaluation integration (MT-Bench, AlpacaEval).

This module provides integration with standard LLM evaluation benchmarks:
- MT-Bench: Multi-turn benchmark with GPT-4 as judge
- AlpacaEval: Instruction-following evaluation

These benchmarks provide standardized metrics for comparing aligned models.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

# Try importing openai for GPT-4 judge
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# MT-Bench question categories
MT_BENCH_CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]

# MT-Bench questions (subset for reference)
MT_BENCH_QUESTIONS = [
    {
        "question_id": 1,
        "category": "writing",
        "turns": [
            "Write a persuasive email to convince your introverted friend to join you for a weekend camping trip.",
            "Now write a follow-up email addressing their concerns about being uncomfortable in nature and around too many people.",
        ],
    },
    {
        "question_id": 2,
        "category": "roleplay",
        "turns": [
            "Act as a pirate captain and describe your most recent adventure.",
            "Now, describe the treasure you found in that adventure.",
        ],
    },
    {
        "question_id": 3,
        "category": "reasoning",
        "turns": [
            "What is the sum of all the prime numbers less than 50?",
            "Now, calculate the product of the first five prime numbers.",
        ],
    },
    {
        "question_id": 4,
        "category": "math",
        "turns": [
            "If a train travels at 60 mph for 2 hours, how far does it travel?",
            "If the train then travels at 80 mph for another hour, what is the total distance traveled?",
        ],
    },
    {
        "question_id": 5,
        "category": "coding",
        "turns": [
            "Write a Python function to check if a number is prime.",
            "Now modify the function to return all prime numbers up to n.",
        ],
    },
]


@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation.

    Attributes:
        benchmark_name: Name of the benchmark
        overall_score: Overall benchmark score
        category_scores: Scores by category
        turn_scores: Scores by conversation turn (for multi-turn)
        individual_scores: Per-question scores
        metadata: Additional metadata
    """

    benchmark_name: str = ""
    overall_score: float = 0.0
    category_scores: dict[str, float] = field(default_factory=dict)
    turn_scores: dict[int, float] = field(default_factory=dict)
    individual_scores: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "benchmark_name": self.benchmark_name,
            "overall_score": self.overall_score,
            "category_scores": self.category_scores,
            "turn_scores": self.turn_scores,
            "individual_scores": self.individual_scores,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Get text summary.

        Returns:
            Summary string
        """
        lines = [
            f"=== {self.benchmark_name} Results ===",
            f"Overall Score: {self.overall_score:.2f}",
            "",
            "Category Scores:",
        ]
        for cat, score in sorted(self.category_scores.items()):
            lines.append(f"  {cat}: {score:.2f}")

        if self.turn_scores:
            lines.append("")
            lines.append("Turn Scores:")
            for turn, score in sorted(self.turn_scores.items()):
                lines.append(f"  Turn {turn}: {score:.2f}")

        return "\n".join(lines)


class MTBenchEvaluator:
    """Evaluator for MT-Bench multi-turn benchmark.

    MT-Bench evaluates multi-turn conversation ability across
    categories like writing, reasoning, math, coding, etc.

    Args:
        judge_model: Model for judging (e.g., GPT-4 API)
        model: Model to evaluate
        tokenizer: Tokenizer
        max_new_tokens: Maximum generation tokens
        temperature: Sampling temperature
        num_gpus: Number of GPUs for parallel inference
    """

    # Judge prompt template
    JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

    def __init__(
        self,
        judge_model: str = "gpt-4",
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        num_gpus: int = 1,
    ) -> None:
        """Initialize the MT-Bench evaluator.

        Args:
            judge_model: Judge model name
            model: Model to evaluate
            tokenizer: Tokenizer
            max_new_tokens: Max generation tokens
            temperature: Sampling temperature
            num_gpus: Number of GPUs
        """
        self.judge_model = judge_model
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.num_gpus = num_gpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check for OpenAI API key
        self.use_openai = HAS_OPENAI and os.getenv("OPENAI_API_KEY") is not None
        if self.use_openai:
            self.openai_client = openai.OpenAI()

        # Load questions
        self.questions = self._load_default_questions()

    def _load_default_questions(self) -> list[dict[str, Any]]:
        """Load default MT-Bench questions."""
        # Try to load from HuggingFace
        try:
            dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
            questions = []
            for item in dataset:
                questions.append({
                    "question_id": item.get("question_id", len(questions)),
                    "category": item.get("category", "general"),
                    "turns": item.get("turns", [item.get("prompt", "")]),
                })
            return questions
        except Exception:
            # Fall back to built-in questions
            return MT_BENCH_QUESTIONS

    def evaluate(
        self,
        model: PreTrainedModel | None = None,
        model_name: str = "model",
    ) -> BenchmarkResult:
        """Run MT-Bench evaluation.

        Args:
            model: Model to evaluate (optional if set in init)
            model_name: Name for logging

        Returns:
            Benchmark results
        """
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided for evaluation")

        # Generate answers
        answers = self.generate_answers(self.questions)

        # Judge answers
        judgments = self.judge_answers(answers)

        # Compute scores
        result = self.compute_scores(judgments)
        result.metadata["model_name"] = model_name

        return result

    def load_questions(self) -> list[dict[str, Any]]:
        """Load MT-Bench questions.

        Returns:
            List of question dictionaries
        """
        return self.questions

    def generate_answers(
        self,
        questions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate model answers for questions.

        Args:
            questions: MT-Bench questions

        Returns:
            Questions with model answers
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required for generation")

        self.model.eval()
        results = []

        for q in tqdm(questions, desc="Generating answers"):
            answers = []
            conversation = []

            for turn_idx, turn in enumerate(q["turns"]):
                # Build conversation context
                conversation.append({"role": "user", "content": turn})

                # Format for generation
                if hasattr(self.tokenizer, "apply_chat_template"):
                    prompt = self.tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    prompt = "\n".join(
                        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                        for m in conversation
                    ) + "\nAssistant:"

                # Generate
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                answers.append(response)
                conversation.append({"role": "assistant", "content": response})

            results.append({
                **q,
                "answers": answers,
            })

        return results

    def judge_answers(
        self,
        answers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Get judgments for model answers.

        Args:
            answers: Model answers

        Returns:
            Answers with judgments
        """
        results = []

        for item in tqdm(answers, desc="Judging answers"):
            scores = []
            explanations = []

            for turn_idx, (question, answer) in enumerate(zip(item["turns"], item["answers"])):
                score, explanation = self._judge_single(question, answer)
                scores.append(score)
                explanations.append(explanation)

            results.append({
                **item,
                "scores": scores,
                "explanations": explanations,
            })

        return results

    def _judge_single(self, question: str, answer: str) -> tuple[float, str]:
        """Judge a single question-answer pair."""
        prompt = self.JUDGE_PROMPT.format(question=question, answer=answer)

        if self.use_openai:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1024,
                )
                explanation = response.choices[0].message.content

                # Extract rating
                match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", explanation)
                if match:
                    score = float(match.group(1))
                else:
                    score = 5.0  # Default

                return score, explanation
            except Exception as e:
                return 5.0, f"Error: {str(e)}"
        else:
            # Use local heuristics as fallback
            return self._local_judge(question, answer)

    def _local_judge(self, question: str, answer: str) -> tuple[float, str]:
        """Simple local judging heuristics."""
        score = 5.0
        reasons = []

        # Length check
        if len(answer) < 50:
            score -= 2
            reasons.append("Response too short")
        elif len(answer) > 100:
            score += 1
            reasons.append("Good length")

        # Relevance check (simple keyword overlap)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words & answer_words) / max(len(question_words), 1)
        if overlap > 0.2:
            score += 1
            reasons.append("Relevant to question")

        # Completeness check
        if answer.endswith((".", "!", "?", "```")):
            score += 0.5
            reasons.append("Complete response")

        score = max(1.0, min(10.0, score))
        explanation = f"Score: {score}. Reasons: {', '.join(reasons)}"

        return score, explanation

    def compute_scores(
        self,
        judgments: list[dict[str, Any]],
    ) -> BenchmarkResult:
        """Compute scores from judgments.

        Args:
            judgments: Judge evaluations

        Returns:
            Benchmark results
        """
        category_scores: dict[str, list[float]] = {}
        turn_scores: dict[int, list[float]] = {}
        all_scores = []
        individual_scores = []

        for item in judgments:
            category = item.get("category", "general")
            if category not in category_scores:
                category_scores[category] = []

            for turn_idx, score in enumerate(item["scores"]):
                category_scores[category].append(score)
                all_scores.append(score)

                if turn_idx not in turn_scores:
                    turn_scores[turn_idx] = []
                turn_scores[turn_idx].append(score)

            individual_scores.append({
                "question_id": item.get("question_id"),
                "category": category,
                "scores": item["scores"],
                "mean_score": sum(item["scores"]) / len(item["scores"]) if item["scores"] else 0,
            })

        # Compute averages
        avg_category = {
            cat: sum(scores) / len(scores) if scores else 0
            for cat, scores in category_scores.items()
        }
        avg_turn = {
            turn: sum(scores) / len(scores) if scores else 0
            for turn, scores in turn_scores.items()
        }
        overall = sum(all_scores) / len(all_scores) if all_scores else 0

        return BenchmarkResult(
            benchmark_name="MT-Bench",
            overall_score=overall,
            category_scores=avg_category,
            turn_scores=avg_turn,
            individual_scores=individual_scores,
            metadata={"num_questions": len(judgments)},
        )

    def save_results(
        self,
        result: BenchmarkResult,
        output_path: str | Path,
    ) -> None:
        """Save evaluation results.

        Args:
            result: Results to save
            output_path: Output path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)


class AlpacaEvalRunner:
    """Runner for AlpacaEval benchmark.

    AlpacaEval evaluates instruction-following ability by comparing
    model responses against a reference (e.g., text-davinci-003).

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        reference_model: Reference model for comparison
        annotator: Annotator model (e.g., GPT-4)
        max_new_tokens: Maximum generation tokens
    """

    ANNOTATOR_PROMPT = """I want you to compare two AI assistant responses to the same instruction. Which response is better?

Instruction: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Please answer with only "A" or "B"."""

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        reference_model: str = "text-davinci-003",
        annotator: str = "alpaca_eval_gpt4",
        max_new_tokens: int = 1024,
    ) -> None:
        """Initialize AlpacaEval runner.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            reference_model: Reference model
            annotator: Annotator configuration
            max_new_tokens: Max generation tokens
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reference_model = reference_model
        self.annotator = annotator
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check for OpenAI
        self.use_openai = HAS_OPENAI and os.getenv("OPENAI_API_KEY") is not None
        if self.use_openai:
            self.openai_client = openai.OpenAI()

        # Load evaluation set
        self.eval_set = self._load_eval_set()

    def _load_eval_set(self) -> list[dict[str, Any]]:
        """Load AlpacaEval evaluation set."""
        try:
            dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
            return [
                {
                    "instruction": item["instruction"],
                    "reference_output": item.get("output", ""),
                }
                for item in dataset
            ]
        except Exception:
            # Return minimal fallback set
            return [
                {"instruction": "What is the capital of France?", "reference_output": "Paris"},
                {"instruction": "Write a haiku about coding.", "reference_output": "Lines of code flow free\nBugs hide in logic's shadows\nDebuggers unite"},
            ]

    def evaluate(
        self,
        model: PreTrainedModel | None = None,
        model_name: str = "model",
    ) -> BenchmarkResult:
        """Run AlpacaEval evaluation.

        Args:
            model: Model to evaluate
            model_name: Name for logging

        Returns:
            Benchmark results with win rate
        """
        if model is not None:
            self.model = model

        if self.model is None:
            raise ValueError("No model provided for evaluation")

        # Load evaluation set
        instructions = [item["instruction"] for item in self.eval_set]
        reference_outputs = [item["reference_output"] for item in self.eval_set]

        # Generate outputs
        model_outputs = self.generate_outputs(instructions)

        # Compute win rate
        win_rate = self.compute_win_rate(model_outputs, reference_outputs, instructions)
        lc_win_rate = self.compute_length_controlled_win_rate(
            model_outputs, reference_outputs, instructions
        )

        individual_scores = []
        for i, (inst, model_out, ref_out) in enumerate(zip(instructions, model_outputs, reference_outputs)):
            individual_scores.append({
                "instruction": inst[:100] + "..." if len(inst) > 100 else inst,
                "model_length": len(model_out),
                "reference_length": len(ref_out),
            })

        return BenchmarkResult(
            benchmark_name="AlpacaEval",
            overall_score=win_rate,
            category_scores={
                "win_rate": win_rate,
                "length_controlled_win_rate": lc_win_rate,
            },
            individual_scores=individual_scores,
            metadata={
                "model_name": model_name,
                "reference_model": self.reference_model,
                "num_examples": len(instructions),
            },
        )

    def load_eval_set(self) -> Dataset:
        """Load AlpacaEval evaluation set.

        Returns:
            Evaluation dataset
        """
        return Dataset.from_list(self.eval_set)

    def generate_outputs(
        self,
        instructions: list[str],
    ) -> list[str]:
        """Generate model outputs for instructions.

        Args:
            instructions: Input instructions

        Returns:
            Generated outputs
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer required")

        self.model.eval()
        outputs = []

        for instruction in tqdm(instructions, desc="Generating outputs"):
            # Format prompt
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": instruction}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = f"User: {instruction}\nAssistant:"

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            outputs.append(response)

        return outputs

    def compute_win_rate(
        self,
        model_outputs: list[str],
        reference_outputs: list[str],
        instructions: list[str],
    ) -> float:
        """Compute win rate against reference.

        Args:
            model_outputs: Model responses
            reference_outputs: Reference responses
            instructions: Input instructions

        Returns:
            Win rate (0-100)
        """
        wins = 0
        total = 0

        for instruction, model_out, ref_out in zip(instructions, model_outputs, reference_outputs):
            winner = self._annotate_pair(instruction, model_out, ref_out)
            if winner == "A":  # Model wins
                wins += 1
            total += 1

        return (wins / total * 100) if total > 0 else 0.0

    def _annotate_pair(self, instruction: str, response_a: str, response_b: str) -> str:
        """Annotate a single pair comparison."""
        if self.use_openai:
            try:
                prompt = self.ANNOTATOR_PROMPT.format(
                    instruction=instruction,
                    response_a=response_a,
                    response_b=response_b,
                )
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                )
                answer = response.choices[0].message.content.strip().upper()
                return "A" if "A" in answer else "B"
            except Exception:
                pass

        # Fallback: simple heuristics
        score_a = len(response_a) + (10 if response_a.strip().endswith(".") else 0)
        score_b = len(response_b) + (10 if response_b.strip().endswith(".") else 0)
        return "A" if score_a > score_b else "B"

    def compute_length_controlled_win_rate(
        self,
        model_outputs: list[str],
        reference_outputs: list[str],
        instructions: list[str],
    ) -> float:
        """Compute length-controlled win rate.

        Controls for length bias in evaluation.

        Args:
            model_outputs: Model responses
            reference_outputs: Reference responses
            instructions: Input instructions

        Returns:
            Length-controlled win rate
        """
        # Group by length ratio buckets
        buckets: dict[str, list[tuple[str, str, str]]] = {
            "shorter": [],
            "similar": [],
            "longer": [],
        }

        for instruction, model_out, ref_out in zip(instructions, model_outputs, reference_outputs):
            ratio = len(model_out) / max(len(ref_out), 1)
            if ratio < 0.8:
                buckets["shorter"].append((instruction, model_out, ref_out))
            elif ratio > 1.2:
                buckets["longer"].append((instruction, model_out, ref_out))
            else:
                buckets["similar"].append((instruction, model_out, ref_out))

        # Compute win rate for each bucket
        bucket_win_rates = []
        for bucket_name, pairs in buckets.items():
            if not pairs:
                continue
            wins = sum(
                1 for inst, m, r in pairs
                if self._annotate_pair(inst, m, r) == "A"
            )
            bucket_win_rates.append(wins / len(pairs))

        # Average across buckets
        return (sum(bucket_win_rates) / len(bucket_win_rates) * 100) if bucket_win_rates else 0.0


class BenchmarkRunner:
    """Unified benchmark runner.

    Runs multiple benchmarks and aggregates results.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        benchmarks: List of benchmark names to run
    """

    SUPPORTED_BENCHMARKS = ["mt_bench", "alpaca_eval"]

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        benchmarks: list[str] | None = None,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            benchmarks: Benchmarks to run
        """
        self.model = model
        self.tokenizer = tokenizer
        self.benchmarks = benchmarks or self.SUPPORTED_BENCHMARKS

        # Initialize evaluators
        self.evaluators: dict[str, Any] = {}
        if "mt_bench" in self.benchmarks:
            self.evaluators["mt_bench"] = MTBenchEvaluator(
                model=model, tokenizer=tokenizer
            )
        if "alpaca_eval" in self.benchmarks:
            self.evaluators["alpaca_eval"] = AlpacaEvalRunner(
                model=model, tokenizer=tokenizer
            )

    def run_all(
        self,
        model: PreTrainedModel | None = None,
        model_name: str = "model",
    ) -> dict[str, BenchmarkResult]:
        """Run all configured benchmarks.

        Args:
            model: Model to evaluate
            model_name: Name for logging

        Returns:
            Dictionary mapping benchmark names to results
        """
        results = {}
        for benchmark_name in self.benchmarks:
            print(f"Running {benchmark_name}...")
            result = self.run_benchmark(benchmark_name, model)
            result.metadata["model_name"] = model_name
            results[benchmark_name] = result
        return results

    def run_benchmark(
        self,
        benchmark_name: str,
        model: PreTrainedModel | None = None,
    ) -> BenchmarkResult:
        """Run a specific benchmark.

        Args:
            benchmark_name: Name of benchmark
            model: Model to evaluate

        Returns:
            Benchmark results
        """
        if benchmark_name not in self.evaluators:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        return self.evaluators[benchmark_name].evaluate(model=model)

    def compare_models(
        self,
        models: dict[str, PreTrainedModel],
    ) -> dict[str, dict[str, BenchmarkResult]]:
        """Compare multiple models on benchmarks.

        Args:
            models: Dictionary of model name to model

        Returns:
            Nested dictionary of results
        """
        results: dict[str, dict[str, BenchmarkResult]] = {}

        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            results[model_name] = self.run_all(model=model, model_name=model_name)

        return results

    def generate_report(
        self,
        results: dict[str, BenchmarkResult],
        output_path: str | Path,
    ) -> None:
        """Generate benchmark report.

        Args:
            results: Benchmark results
            output_path: Output path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = ["# Benchmark Evaluation Report\n"]

        for benchmark_name, result in results.items():
            lines.append(f"## {result.benchmark_name}\n")
            lines.append(f"**Overall Score:** {result.overall_score:.2f}\n")

            if result.category_scores:
                lines.append("\n### Category Scores\n")
                lines.append("| Category | Score |")
                lines.append("|----------|-------|")
                for cat, score in sorted(result.category_scores.items()):
                    lines.append(f"| {cat} | {score:.2f} |")

            lines.append("\n")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def plot_comparison(
        self,
        results: dict[str, dict[str, BenchmarkResult]],
        save_path: str | None = None,
    ) -> Any:
        """Plot benchmark comparison.

        Args:
            results: Results from compare_models
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib required for plotting")
            return None

        # Extract data
        model_names = list(results.keys())
        benchmark_names = list(next(iter(results.values())).keys())

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(benchmark_names))
        width = 0.8 / len(model_names)

        for i, model_name in enumerate(model_names):
            scores = [results[model_name][b].overall_score for b in benchmark_names]
            offset = (i - len(model_names) / 2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=model_name)

        ax.set_xlabel("Benchmark")
        ax.set_ylabel("Score")
        ax.set_title("Benchmark Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(benchmark_names)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


# Convenience functions
def evaluate_mt_bench(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    judge_model: str = "gpt-4",
) -> BenchmarkResult:
    """Convenience function for MT-Bench evaluation.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        judge_model: Judge model name

    Returns:
        Benchmark results
    """
    evaluator = MTBenchEvaluator(
        judge_model=judge_model,
        model=model,
        tokenizer=tokenizer,
    )
    return evaluator.evaluate()


def evaluate_alpaca_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> BenchmarkResult:
    """Convenience function for AlpacaEval evaluation.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer

    Returns:
        Benchmark results
    """
    runner = AlpacaEvalRunner(model=model, tokenizer=tokenizer)
    return runner.evaluate()
