"""
Human preference evaluation framework.

This module provides tools for collecting and analyzing human
preference evaluations:

- Annotation task management
- Inter-annotator agreement
- Preference aggregation
- Quality control

Human evaluation is the gold standard for alignment assessment.
"""

import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class AnnotationType(Enum):
    """Types of annotation tasks."""

    PAIRWISE = "pairwise"
    LIKERT = "likert"
    RANKING = "ranking"
    BINARY = "binary"


class QualityDimension(Enum):
    """Dimensions of response quality."""

    HELPFULNESS = "helpfulness"
    HARMLESSNESS = "harmlessness"
    HONESTY = "honesty"
    ACCURACY = "accuracy"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"


@dataclass
class AnnotationTask:
    """A single annotation task.

    Attributes:
        task_id: Unique task identifier
        task_type: Type of annotation
        prompt: Input prompt
        responses: Response(s) to evaluate
        dimensions: Quality dimensions to rate
        metadata: Additional metadata
    """

    task_id: str = ""
    task_type: AnnotationType = AnnotationType.PAIRWISE
    prompt: str = ""
    responses: list[str] = field(default_factory=list)
    dimensions: list[QualityDimension] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "prompt": self.prompt,
            "responses": self.responses,
            "dimensions": [d.value for d in self.dimensions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationTask":
        """Create from dictionary."""
        return cls(
            task_id=data.get("task_id", ""),
            task_type=AnnotationType(data.get("task_type", "pairwise")),
            prompt=data.get("prompt", ""),
            responses=data.get("responses", []),
            dimensions=[QualityDimension(d) for d in data.get("dimensions", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Annotation:
    """A completed annotation.

    Attributes:
        task_id: Task identifier
        annotator_id: Annotator identifier
        preference: Selected preference (for pairwise)
        ratings: Ratings by dimension (for Likert)
        ranking: Response ranking
        comments: Optional annotator comments
        time_spent: Time spent on annotation
    """

    task_id: str = ""
    annotator_id: str = ""
    preference: int | None = None
    ratings: dict[QualityDimension, int] = field(default_factory=dict)
    ranking: list[int] = field(default_factory=list)
    comments: str = ""
    time_spent: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "annotator_id": self.annotator_id,
            "preference": self.preference,
            "ratings": {k.value: v for k, v in self.ratings.items()},
            "ranking": self.ranking,
            "comments": self.comments,
            "time_spent": self.time_spent,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Annotation":
        """Create from dictionary."""
        return cls(
            task_id=data.get("task_id", ""),
            annotator_id=data.get("annotator_id", ""),
            preference=data.get("preference"),
            ratings={QualityDimension(k): v for k, v in data.get("ratings", {}).items()},
            ranking=data.get("ranking", []),
            comments=data.get("comments", ""),
            time_spent=data.get("time_spent", 0.0),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class EvaluationResult:
    """Result from human evaluation.

    Attributes:
        num_tasks: Number of tasks evaluated
        num_annotators: Number of annotators
        agreement_scores: Inter-annotator agreement
        preference_rates: Win rates for each model
        dimension_scores: Scores by quality dimension
        confidence_intervals: Confidence intervals
        quality_metrics: Annotation quality metrics
    """

    num_tasks: int = 0
    num_annotators: int = 0
    agreement_scores: dict[str, float] = field(default_factory=dict)
    preference_rates: dict[str, float] = field(default_factory=dict)
    dimension_scores: dict[QualityDimension, dict[str, float]] = field(
        default_factory=dict
    )
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    quality_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "num_tasks": self.num_tasks,
            "num_annotators": self.num_annotators,
            "agreement_scores": self.agreement_scores,
            "preference_rates": self.preference_rates,
            "dimension_scores": {
                k.value: v for k, v in self.dimension_scores.items()
            },
            "confidence_intervals": {
                k: list(v) for k, v in self.confidence_intervals.items()
            },
            "quality_metrics": self.quality_metrics,
        }

    def summary(self) -> str:
        """Get text summary.

        Returns:
            Summary string
        """
        lines = [
            "=== Human Evaluation Results ===",
            f"Tasks: {self.num_tasks}",
            f"Annotators: {self.num_annotators}",
            "",
            "Agreement Scores:",
        ]

        for metric, score in self.agreement_scores.items():
            lines.append(f"  {metric}: {score:.3f}")

        if self.preference_rates:
            lines.append("")
            lines.append("Preference Rates:")
            for model, rate in self.preference_rates.items():
                ci = self.confidence_intervals.get(model, (0, 0))
                lines.append(f"  {model}: {rate:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]")

        if self.quality_metrics:
            lines.append("")
            lines.append("Quality Metrics:")
            for metric, value in self.quality_metrics.items():
                lines.append(f"  {metric}: {value:.3f}")

        return "\n".join(lines)


class PreferenceCollector:
    """Collector for human preferences.

    Manages the annotation collection process.

    Args:
        task_type: Type of annotation task
        dimensions: Quality dimensions to evaluate
        num_annotators_per_task: Annotators per task
    """

    def __init__(
        self,
        task_type: AnnotationType = AnnotationType.PAIRWISE,
        dimensions: list[QualityDimension] | None = None,
        num_annotators_per_task: int = 3,
    ) -> None:
        """Initialize the collector.

        Args:
            task_type: Annotation type
            dimensions: Quality dimensions
            num_annotators_per_task: Annotators per task
        """
        self.task_type = task_type
        self.dimensions = dimensions or [
            QualityDimension.HELPFULNESS,
            QualityDimension.ACCURACY,
            QualityDimension.COHERENCE,
        ]
        self.num_annotators_per_task = num_annotators_per_task
        self.tasks: list[AnnotationTask] = []
        self.annotations: list[Annotation] = []

    def create_tasks(
        self,
        prompts: list[str],
        responses_a: list[str],
        responses_b: list[str] | None = None,
        model_names: tuple[str, str] = ("A", "B"),
    ) -> list[AnnotationTask]:
        """Create annotation tasks.

        Args:
            prompts: Input prompts
            responses_a: Responses from model A
            responses_b: Responses from model B (pairwise)
            model_names: Names for models

        Returns:
            List of annotation tasks
        """
        tasks = []

        for i, prompt in enumerate(prompts):
            if self.task_type == AnnotationType.PAIRWISE and responses_b is not None:
                # Pairwise comparison
                responses = [responses_a[i], responses_b[i]]

                # Randomly shuffle to avoid position bias
                if np.random.random() > 0.5:
                    responses = responses[::-1]
                    shuffled = True
                else:
                    shuffled = False

                task = AnnotationTask(
                    task_type=self.task_type,
                    prompt=prompt,
                    responses=responses,
                    dimensions=self.dimensions,
                    metadata={
                        "model_names": model_names,
                        "shuffled": shuffled,
                        "original_index": i,
                    },
                )
            else:
                # Single response evaluation (Likert, etc.)
                task = AnnotationTask(
                    task_type=self.task_type,
                    prompt=prompt,
                    responses=[responses_a[i]],
                    dimensions=self.dimensions,
                    metadata={"original_index": i},
                )

            tasks.append(task)

        self.tasks.extend(tasks)
        return tasks

    def create_tasks_from_models(
        self,
        model_a: PreTrainedModel,
        model_b: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompts: list[str],
    ) -> list[AnnotationTask]:
        """Create tasks by generating from models.

        Args:
            model_a: First model
            model_b: Second model
            tokenizer: Tokenizer
            prompts: Input prompts

        Returns:
            Annotation tasks
        """
        # Generate from both models
        responses_a = self._generate_responses(model_a, tokenizer, prompts)
        responses_b = self._generate_responses(model_b, tokenizer, prompts)

        return self.create_tasks(prompts, responses_a, responses_b)

    def _generate_responses(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompts: list[str],
    ) -> list[str]:
        """Generate responses from a model."""
        device = next(model.parameters()).device
        model.eval()
        responses = []

        for prompt in tqdm(prompts, desc="Generating"):
            if hasattr(tokenizer, "apply_chat_template"):
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                formatted = f"User: {prompt}\nAssistant:"

            inputs = tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)

        return responses

    def export_tasks(
        self,
        tasks: list[AnnotationTask],
        output_path: str | Path,
        format: str = "json",
    ) -> None:
        """Export tasks for annotation platform.

        Args:
            tasks: Tasks to export
            output_path: Output path
            format: Export format
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = {
                "metadata": {
                    "task_type": tasks[0].task_type.value if tasks else "",
                    "num_tasks": len(tasks),
                    "exported_at": datetime.now().isoformat(),
                },
                "tasks": [t.to_dict() for t in tasks],
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            import csv

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["task_id", "prompt", "response_a", "response_b"])
                for task in tasks:
                    row = [task.task_id, task.prompt]
                    row.extend(task.responses[:2] if len(task.responses) >= 2 else task.responses + [""])
                    writer.writerow(row)

        elif format == "jsonl":
            with open(output_path, "w") as f:
                for task in tasks:
                    f.write(json.dumps(task.to_dict()) + "\n")

    def import_annotations(
        self,
        annotations_path: str | Path,
    ) -> list[Annotation]:
        """Import completed annotations.

        Args:
            annotations_path: Path to annotations

        Returns:
            List of annotations
        """
        annotations_path = Path(annotations_path)

        if annotations_path.suffix == ".json":
            with open(annotations_path) as f:
                data = json.load(f)

            if isinstance(data, list):
                annotations = [Annotation.from_dict(a) for a in data]
            else:
                annotations = [Annotation.from_dict(a) for a in data.get("annotations", [])]

        elif annotations_path.suffix == ".jsonl":
            annotations = []
            with open(annotations_path) as f:
                for line in f:
                    annotations.append(Annotation.from_dict(json.loads(line)))

        else:
            raise ValueError(f"Unsupported format: {annotations_path.suffix}")

        self.annotations.extend(annotations)
        return annotations


class HumanEvalFramework:
    """Framework for human preference evaluation.

    Coordinates the full evaluation pipeline from task creation
    to result analysis.

    Args:
        collector: Preference collector
        dimensions: Quality dimensions to evaluate
    """

    def __init__(
        self,
        collector: PreferenceCollector | None = None,
        dimensions: list[QualityDimension] | None = None,
    ) -> None:
        """Initialize the framework.

        Args:
            collector: Preference collector
            dimensions: Quality dimensions
        """
        self.collector = collector or PreferenceCollector()
        self.dimensions = dimensions or [
            QualityDimension.HELPFULNESS,
            QualityDimension.ACCURACY,
            QualityDimension.COHERENCE,
        ]

    def evaluate(
        self,
        tasks: list[AnnotationTask],
        annotations: list[Annotation],
    ) -> EvaluationResult:
        """Analyze collected annotations.

        Args:
            tasks: Annotation tasks
            annotations: Collected annotations

        Returns:
            Evaluation results
        """
        # Build task lookup
        task_lookup = {t.task_id: t for t in tasks}

        # Group annotations by task
        annotations_by_task: dict[str, list[Annotation]] = defaultdict(list)
        for ann in annotations:
            annotations_by_task[ann.task_id].append(ann)

        # Compute metrics
        result = EvaluationResult(
            num_tasks=len(tasks),
            num_annotators=len(set(a.annotator_id for a in annotations)),
        )

        # Agreement scores
        result.agreement_scores = self.compute_agreement(annotations)

        # Aggregate preferences
        aggregated = self.aggregate_preferences(annotations)

        # Compute win rates
        result.preference_rates = self.compute_win_rates(aggregated)

        # Compute confidence intervals
        preferences_list = list(aggregated.values())
        if preferences_list:
            result.confidence_intervals["model_a"] = self.compute_confidence_intervals(
                [1 if p == 0 else 0 for p in preferences_list]
            )
            result.confidence_intervals["model_b"] = self.compute_confidence_intervals(
                [1 if p == 1 else 0 for p in preferences_list]
            )

        # Compute dimension scores
        result.dimension_scores = self._compute_dimension_scores(annotations)

        # Quality metrics
        result.quality_metrics = self.check_annotation_quality(annotations)

        return result

    def compute_agreement(
        self,
        annotations: list[Annotation],
    ) -> dict[str, float]:
        """Compute inter-annotator agreement.

        Args:
            annotations: All annotations

        Returns:
            Agreement metrics (Cohen's kappa, Fleiss' kappa, etc.)
        """
        # Group by task
        by_task: dict[str, list[int]] = defaultdict(list)
        for ann in annotations:
            if ann.preference is not None:
                by_task[ann.task_id].append(ann.preference)

        if not by_task:
            return {"agreement_rate": 0.0}

        # Simple agreement rate
        agreements = 0
        total = 0
        for task_id, prefs in by_task.items():
            if len(prefs) >= 2:
                # Check if annotators agree
                most_common = max(set(prefs), key=prefs.count)
                agreements += sum(1 for p in prefs if p == most_common)
                total += len(prefs)

        agreement_rate = agreements / total if total > 0 else 0.0

        # Compute Fleiss' kappa (simplified)
        kappa = self._compute_fleiss_kappa(by_task)

        return {
            "agreement_rate": agreement_rate,
            "fleiss_kappa": kappa,
        }

    def _compute_fleiss_kappa(self, by_task: dict[str, list[int]]) -> float:
        """Compute Fleiss' kappa for inter-rater reliability."""
        if not by_task:
            return 0.0

        # Convert to matrix format
        n_items = len(by_task)
        categories = {0, 1}  # Binary preference
        n_categories = len(categories)

        # Count ratings per category for each item
        ratings_matrix = []
        n_raters = 0

        for task_id, prefs in by_task.items():
            if prefs:
                n_raters = max(n_raters, len(prefs))
                counts = [prefs.count(c) for c in range(n_categories)]
                ratings_matrix.append(counts)

        if not ratings_matrix or n_raters < 2:
            return 0.0

        # Compute Fleiss' kappa
        # P_i for each item
        n = n_raters
        p_i = []
        for counts in ratings_matrix:
            sum_squared = sum(c * c for c in counts)
            p_i.append((sum_squared - n) / (n * (n - 1)) if n > 1 else 0)

        P_bar = np.mean(p_i) if p_i else 0

        # P_j for each category
        total_ratings = sum(sum(counts) for counts in ratings_matrix)
        if total_ratings == 0:
            return 0.0

        p_j = []
        for j in range(n_categories):
            category_sum = sum(counts[j] for counts in ratings_matrix)
            p_j.append(category_sum / total_ratings)

        P_e = sum(p * p for p in p_j)

        # Kappa
        if P_e == 1:
            return 1.0

        kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0
        return float(kappa)

    def aggregate_preferences(
        self,
        annotations: list[Annotation],
        method: str = "majority",
    ) -> dict[str, int]:
        """Aggregate preferences across annotators.

        Args:
            annotations: All annotations
            method: Aggregation method

        Returns:
            Aggregated preferences by task
        """
        by_task: dict[str, list[int]] = defaultdict(list)
        for ann in annotations:
            if ann.preference is not None:
                by_task[ann.task_id].append(ann.preference)

        aggregated = {}

        for task_id, prefs in by_task.items():
            if not prefs:
                continue

            if method == "majority":
                # Simple majority vote
                aggregated[task_id] = max(set(prefs), key=prefs.count)

            elif method == "unanimous":
                # Only include if unanimous
                if len(set(prefs)) == 1:
                    aggregated[task_id] = prefs[0]

            elif method == "weighted":
                # Could weight by annotator quality
                # For now, same as majority
                aggregated[task_id] = max(set(prefs), key=prefs.count)

        return aggregated

    def compute_win_rates(
        self,
        aggregated_preferences: dict[str, int],
    ) -> dict[str, float]:
        """Compute win rates from preferences.

        Args:
            aggregated_preferences: Aggregated preferences

        Returns:
            Win rates for each model
        """
        if not aggregated_preferences:
            return {"model_a": 0.0, "model_b": 0.0, "tie": 0.0}

        total = len(aggregated_preferences)
        model_a_wins = sum(1 for p in aggregated_preferences.values() if p == 0)
        model_b_wins = sum(1 for p in aggregated_preferences.values() if p == 1)
        ties = sum(1 for p in aggregated_preferences.values() if p == 2)

        return {
            "model_a": model_a_wins / total,
            "model_b": model_b_wins / total,
            "tie": ties / total,
        }

    def compute_confidence_intervals(
        self,
        preferences: list[int],
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Compute confidence intervals for win rate.

        Args:
            preferences: List of preference values
            confidence: Confidence level

        Returns:
            (lower, upper) confidence interval
        """
        if not preferences:
            return (0.0, 0.0)

        n = len(preferences)
        p = np.mean(preferences)

        # Wilson score interval
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)

        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        spread = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator

        lower = max(0, center - spread)
        upper = min(1, center + spread)

        return (lower, upper)

    def _compute_dimension_scores(
        self,
        annotations: list[Annotation],
    ) -> dict[QualityDimension, dict[str, float]]:
        """Compute scores by quality dimension."""
        dimension_scores: dict[QualityDimension, list[int]] = defaultdict(list)

        for ann in annotations:
            for dim, rating in ann.ratings.items():
                dimension_scores[dim].append(rating)

        result = {}
        for dim, ratings in dimension_scores.items():
            if ratings:
                result[dim] = {
                    "mean": np.mean(ratings),
                    "std": np.std(ratings),
                    "min": min(ratings),
                    "max": max(ratings),
                }

        return result

    def check_annotation_quality(
        self,
        annotations: list[Annotation],
    ) -> dict[str, float]:
        """Check annotation quality.

        Includes attention check pass rate, time analysis, etc.

        Args:
            annotations: All annotations

        Returns:
            Quality metrics
        """
        if not annotations:
            return {}

        # Time analysis
        times = [a.time_spent for a in annotations if a.time_spent > 0]
        avg_time = np.mean(times) if times else 0
        std_time = np.std(times) if times else 0

        # Too fast annotations (potential low quality)
        too_fast = sum(1 for t in times if t < 5) / len(times) if times else 0

        # Too slow annotations (potential disengagement)
        too_slow = sum(1 for t in times if t > 300) / len(times) if times else 0

        # Comment rate
        comment_rate = sum(1 for a in annotations if a.comments.strip()) / len(annotations)

        return {
            "avg_time_seconds": avg_time,
            "std_time_seconds": std_time,
            "too_fast_rate": too_fast,
            "too_slow_rate": too_slow,
            "comment_rate": comment_rate,
        }

    def filter_low_quality(
        self,
        annotations: list[Annotation],
        min_time: float = 5.0,
        max_agreement_deviation: float = 0.5,
    ) -> list[Annotation]:
        """Filter low-quality annotations.

        Args:
            annotations: All annotations
            min_time: Minimum time spent
            max_agreement_deviation: Max deviation from majority

        Returns:
            Filtered annotations
        """
        # Group by task for agreement check
        by_task: dict[str, list[Annotation]] = defaultdict(list)
        for ann in annotations:
            by_task[ann.task_id].append(ann)

        # Compute majority for each task
        task_majority = {}
        for task_id, anns in by_task.items():
            prefs = [a.preference for a in anns if a.preference is not None]
            if prefs:
                task_majority[task_id] = max(set(prefs), key=prefs.count)

        filtered = []
        for ann in annotations:
            # Time check
            if ann.time_spent < min_time and ann.time_spent > 0:
                continue

            # Agreement check
            if ann.task_id in task_majority and ann.preference is not None:
                majority = task_majority[ann.task_id]
                task_anns = by_task[ann.task_id]
                agreement_rate = sum(
                    1 for a in task_anns if a.preference == majority
                ) / len(task_anns)

                if ann.preference != majority and agreement_rate > (1 - max_agreement_deviation):
                    continue

            filtered.append(ann)

        return filtered

    def generate_report(
        self,
        result: EvaluationResult,
        output_path: str | Path,
    ) -> None:
        """Generate evaluation report.

        Args:
            result: Evaluation results
            output_path: Output path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Human Evaluation Report\n",
            f"Generated: {datetime.now().isoformat()}\n",
            "\n## Overview\n",
            f"- Tasks Evaluated: {result.num_tasks}",
            f"- Number of Annotators: {result.num_annotators}",
            "\n## Agreement Scores\n",
        ]

        for metric, score in result.agreement_scores.items():
            interpretation = ""
            if metric == "fleiss_kappa":
                if score > 0.8:
                    interpretation = " (Almost perfect)"
                elif score > 0.6:
                    interpretation = " (Substantial)"
                elif score > 0.4:
                    interpretation = " (Moderate)"
                elif score > 0.2:
                    interpretation = " (Fair)"
                else:
                    interpretation = " (Slight)"
            lines.append(f"- {metric}: {score:.3f}{interpretation}")

        lines.append("\n## Preference Rates\n")
        lines.append("| Model | Win Rate | 95% CI |")
        lines.append("|-------|----------|--------|")
        for model, rate in result.preference_rates.items():
            ci = result.confidence_intervals.get(model, (0, 0))
            lines.append(f"| {model} | {rate:.1%} | [{ci[0]:.1%}, {ci[1]:.1%}] |")

        if result.dimension_scores:
            lines.append("\n## Dimension Scores\n")
            lines.append("| Dimension | Mean | Std |")
            lines.append("|-----------|------|-----|")
            for dim, scores in result.dimension_scores.items():
                lines.append(f"| {dim.value} | {scores['mean']:.2f} | {scores['std']:.2f} |")

        lines.append("\n## Quality Metrics\n")
        for metric, value in result.quality_metrics.items():
            lines.append(f"- {metric}: {value:.3f}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def plot_results(
        self,
        result: EvaluationResult,
        save_path: str | None = None,
    ) -> Any:
        """Plot evaluation results.

        Args:
            result: Evaluation results
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Win rates
        ax1 = axes[0]
        models = list(result.preference_rates.keys())
        rates = list(result.preference_rates.values())

        colors = ["#2ecc71", "#e74c3c", "#95a5a6"][:len(models)]
        ax1.bar(models, rates, color=colors)

        # Add confidence intervals
        for i, model in enumerate(models):
            if model in result.confidence_intervals:
                ci = result.confidence_intervals[model]
                ax1.errorbar(
                    i, rates[i],
                    yerr=[[rates[i] - ci[0]], [ci[1] - rates[i]]],
                    fmt="none",
                    color="black",
                    capsize=5,
                )

        ax1.set_ylabel("Win Rate")
        ax1.set_title("Model Preference Rates")
        ax1.set_ylim(0, 1)

        # Dimension scores
        ax2 = axes[1]
        if result.dimension_scores:
            dims = [d.value for d in result.dimension_scores.keys()]
            means = [s["mean"] for s in result.dimension_scores.values()]
            stds = [s["std"] for s in result.dimension_scores.values()]

            ax2.barh(dims, means, xerr=stds, capsize=5, color="#3498db")
            ax2.set_xlabel("Score")
            ax2.set_title("Quality Dimension Scores")
        else:
            ax2.text(0.5, 0.5, "No dimension scores", ha="center", va="center")
            ax2.set_title("Quality Dimension Scores")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


# Convenience functions
def create_pairwise_evaluation(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    output_dir: str | Path,
) -> list[AnnotationTask]:
    """Create pairwise evaluation tasks.

    Args:
        model_a: First model
        model_b: Second model
        tokenizer: Tokenizer
        prompts: Evaluation prompts
        output_dir: Output directory

    Returns:
        Created tasks
    """
    collector = PreferenceCollector(task_type=AnnotationType.PAIRWISE)
    tasks = collector.create_tasks_from_models(model_a, model_b, tokenizer, prompts)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collector.export_tasks(tasks, output_dir / "tasks.json")

    return tasks


def analyze_human_eval(
    tasks_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path | None = None,
) -> EvaluationResult:
    """Analyze human evaluation results.

    Args:
        tasks_path: Path to tasks file
        annotations_path: Path to annotations file
        output_path: Optional report output path

    Returns:
        Evaluation results
    """
    # Load tasks
    with open(tasks_path) as f:
        data = json.load(f)
        tasks = [AnnotationTask.from_dict(t) for t in data.get("tasks", data)]

    # Load annotations
    collector = PreferenceCollector()
    annotations = collector.import_annotations(annotations_path)

    # Analyze
    framework = HumanEvalFramework()
    result = framework.evaluate(tasks, annotations)

    # Generate report
    if output_path:
        framework.generate_report(result, output_path)

    return result
