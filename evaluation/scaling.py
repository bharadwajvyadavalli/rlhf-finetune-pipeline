"""
Scaling analysis across model sizes.

This module provides tools for analyzing how alignment quality
scales with model size. Key analyses:

- Reward scaling: How reward improves with model size
- Capability scaling: Task performance across sizes
- Alignment tax: Performance drop from alignment
- Emergent behaviors: Capabilities that appear at scale

Understanding scaling is crucial for planning alignment strategies.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from scipy import optimize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from reward.reward_model import RewardModel


# Model size configurations
PYTHIA_SIZES = {
    70_000_000: "EleutherAI/pythia-70m",
    160_000_000: "EleutherAI/pythia-160m",
    410_000_000: "EleutherAI/pythia-410m",
    1_000_000_000: "EleutherAI/pythia-1b",
    1_400_000_000: "EleutherAI/pythia-1.4b",
    2_800_000_000: "EleutherAI/pythia-2.8b",
}


@dataclass
class ScalingResult:
    """Result from scaling analysis.

    Attributes:
        model_sizes: List of model sizes analyzed
        metrics: Dictionary mapping metrics to per-size values
        scaling_coefficients: Fitted scaling law coefficients
        predictions: Predicted values for larger sizes
        analysis_notes: Analysis observations
    """

    model_sizes: list[int] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    scaling_coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    predictions: dict[str, dict[int, float]] = field(default_factory=dict)
    analysis_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "model_sizes": self.model_sizes,
            "metrics": self.metrics,
            "scaling_coefficients": self.scaling_coefficients,
            "predictions": {k: {str(sk): sv for sk, sv in v.items()} for k, v in self.predictions.items()},
            "analysis_notes": self.analysis_notes,
        }

    def get_metric_by_size(
        self, metric: str, size: int
    ) -> float | None:
        """Get metric value for a specific size.

        Args:
            metric: Metric name
            size: Model size

        Returns:
            Metric value or None
        """
        if metric not in self.metrics:
            return None
        try:
            idx = self.model_sizes.index(size)
            return self.metrics[metric][idx]
        except (ValueError, IndexError):
            return None


class ScalingAnalyzer:
    """Analyzer for scaling behavior across model sizes.

    Evaluates multiple model sizes and fits scaling laws to
    understand how alignment quality scales.

    Args:
        model_family: Model family (e.g., "pythia")
        model_sizes: List of model sizes to analyze
        reward_model: Optional reward model for scoring
        tokenizer: Shared tokenizer
    """

    def __init__(
        self,
        model_family: str = "pythia",
        model_sizes: list[int] | None = None,
        reward_model: RewardModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            model_family: Model family name
            model_sizes: Sizes to analyze
            reward_model: Reward model
            tokenizer: Tokenizer
        """
        self.model_family = model_family
        self.model_sizes = model_sizes or [70_000_000, 410_000_000, 1_000_000_000]
        self.reward_model = reward_model
        self.tokenizer = tokenizer

        # Get model name mapping
        if model_family.lower() == "pythia":
            self.model_names = PYTHIA_SIZES
        else:
            self.model_names = {}

    def analyze(
        self,
        models: dict[int, PreTrainedModel],
        eval_dataset: Dataset,
    ) -> ScalingResult:
        """Run scaling analysis on models.

        Args:
            models: Dictionary mapping size to model
            eval_dataset: Evaluation dataset

        Returns:
            Scaling results
        """
        result = ScalingResult(
            model_sizes=sorted(models.keys()),
            metrics={},
            scaling_coefficients={},
            predictions={},
            analysis_notes=[],
        )

        # Extract prompts from dataset
        prompt_column = "prompt" if "prompt" in eval_dataset.column_names else eval_dataset.column_names[0]
        prompts = eval_dataset[prompt_column][:100]  # Limit for efficiency

        # Compute reward scaling
        print("Computing reward scaling...")
        reward_metrics = self.compute_reward_scaling(models, prompts)

        for size in result.model_sizes:
            if size in reward_metrics:
                for metric, value in reward_metrics[size].items():
                    if metric not in result.metrics:
                        result.metrics[metric] = []
                    result.metrics[metric].append(value)

        # Compute capability scaling
        print("Computing capability scaling...")
        capability_metrics = self.compute_capability_scaling(models, eval_dataset)

        for size in result.model_sizes:
            if size in capability_metrics:
                for metric, value in capability_metrics[size].items():
                    if metric not in result.metrics:
                        result.metrics[metric] = []
                    result.metrics[metric].append(value)

        # Fit scaling laws
        print("Fitting scaling laws...")
        for metric, values in result.metrics.items():
            if len(values) >= 3:  # Need at least 3 points
                try:
                    coefficients = self.fit_scaling_law(result.model_sizes, values)
                    result.scaling_coefficients[metric] = coefficients

                    # Predict for larger sizes
                    predictions = {}
                    for target_size in [2_000_000_000, 7_000_000_000, 13_000_000_000]:
                        pred = self.predict_at_scale(coefficients, target_size)
                        predictions[target_size] = pred
                    result.predictions[metric] = predictions
                except Exception as e:
                    result.analysis_notes.append(f"Could not fit scaling law for {metric}: {str(e)}")

        # Add analysis notes
        self._add_analysis_notes(result)

        return result

    def analyze_from_paths(
        self,
        model_paths: dict[int, str],
        eval_dataset: Dataset,
    ) -> ScalingResult:
        """Run analysis loading models from paths.

        Args:
            model_paths: Dictionary mapping size to path
            eval_dataset: Evaluation dataset

        Returns:
            Scaling results
        """
        models = {}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for size, path in tqdm(model_paths.items(), desc="Loading models"):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                    device_map="auto" if device.type == "cuda" else None,
                )
                if device.type == "cpu":
                    model = model.to(device)
                models[size] = model
            except Exception as e:
                print(f"Failed to load model {path}: {e}")

        if not models:
            raise ValueError("No models loaded successfully")

        result = self.analyze(models, eval_dataset)

        # Clean up
        for model in models.values():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def compute_reward_scaling(
        self,
        models: dict[int, PreTrainedModel],
        prompts: list[str],
    ) -> dict[int, dict[str, float]]:
        """Compute reward metrics across model sizes.

        Args:
            models: Models by size
            prompts: Evaluation prompts

        Returns:
            Reward metrics by size
        """
        results = {}

        for size, model in tqdm(models.items(), desc="Computing rewards"):
            # Generate responses
            responses = self._generate_responses(model, prompts)

            # Compute metrics
            metrics = {
                "mean_response_length": np.mean([len(r.split()) for r in responses]),
                "std_response_length": np.std([len(r.split()) for r in responses]),
            }

            # Compute rewards if reward model available
            if self.reward_model is not None:
                texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
                rewards = self.reward_model.compute_rewards(texts)
                metrics["mean_reward"] = float(rewards.mean())
                metrics["std_reward"] = float(rewards.std())
                metrics["max_reward"] = float(rewards.max())
                metrics["min_reward"] = float(rewards.min())

            results[size] = metrics

        return results

    def _generate_responses(
        self,
        model: PreTrainedModel,
        prompts: list[str],
        max_new_tokens: int = 256,
    ) -> list[str]:
        """Generate responses from a model."""
        device = next(model.parameters()).device
        model.eval()
        responses = []

        # Use tokenizer
        tokenizer = self.tokenizer
        if tokenizer is None:
            # Try to get tokenizer from model
            try:
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            except Exception:
                raise ValueError("Tokenizer required for generation")

        for prompt in prompts:
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
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
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

    def compute_capability_scaling(
        self,
        models: dict[int, PreTrainedModel],
        eval_dataset: Dataset,
    ) -> dict[int, dict[str, float]]:
        """Compute capability metrics across sizes.

        Args:
            models: Models by size
            eval_dataset: Evaluation dataset

        Returns:
            Capability metrics by size
        """
        results = {}

        # Get prompts
        prompt_column = "prompt" if "prompt" in eval_dataset.column_names else eval_dataset.column_names[0]
        prompts = eval_dataset[prompt_column][:50]

        for size, model in tqdm(models.items(), desc="Computing capabilities"):
            # Compute perplexity on evaluation set
            perplexity = self._compute_perplexity(model, prompts)

            # Compute coherence (simple heuristic)
            responses = self._generate_responses(model, prompts[:20])
            coherence = self._compute_coherence(responses)

            results[size] = {
                "perplexity": perplexity,
                "coherence": coherence,
                "completion_rate": sum(1 for r in responses if len(r.strip()) > 10) / len(responses),
            }

        return results

    def _compute_perplexity(self, model: PreTrainedModel, texts: list[str]) -> float:
        """Compute perplexity on texts."""
        device = next(model.parameters()).device
        model.eval()
        tokenizer = self.tokenizer

        if tokenizer is None:
            return float("inf")

        total_loss = 0.0
        total_tokens = 0

        for text in texts[:20]:  # Limit for efficiency
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = np.exp(avg_loss)

        return float(perplexity)

    def _compute_coherence(self, responses: list[str]) -> float:
        """Compute simple coherence metric."""
        coherence_scores = []

        for response in responses:
            # Simple heuristics for coherence
            score = 0.0

            # Length check
            words = response.split()
            if 10 <= len(words) <= 500:
                score += 0.3

            # Sentence structure
            sentences = response.split(".")
            if len(sentences) >= 2:
                score += 0.3

            # No excessive repetition
            unique_words = len(set(words))
            if len(words) > 0 and unique_words / len(words) > 0.5:
                score += 0.4

            coherence_scores.append(score)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def compute_alignment_tax(
        self,
        base_models: dict[int, PreTrainedModel],
        aligned_models: dict[int, PreTrainedModel],
        eval_dataset: Dataset,
    ) -> dict[int, dict[str, float]]:
        """Compute alignment tax across sizes.

        Alignment tax is the performance drop on capabilities
        after alignment.

        Args:
            base_models: Pre-alignment models
            aligned_models: Post-alignment models
            eval_dataset: Capability evaluation dataset

        Returns:
            Alignment tax by size
        """
        results = {}

        # Compute capabilities for both
        base_caps = self.compute_capability_scaling(base_models, eval_dataset)
        aligned_caps = self.compute_capability_scaling(aligned_models, eval_dataset)

        for size in base_models.keys():
            if size not in aligned_caps:
                continue

            base = base_caps[size]
            aligned = aligned_caps[size]

            tax = {}
            for metric in base.keys():
                if metric in aligned:
                    # Tax is the relative change (negative means degradation)
                    if base[metric] != 0:
                        if metric == "perplexity":
                            # For perplexity, lower is better, so increase is bad
                            tax[f"{metric}_tax"] = (aligned[metric] - base[metric]) / base[metric]
                        else:
                            # For other metrics, higher is better, so decrease is bad
                            tax[f"{metric}_tax"] = (base[metric] - aligned[metric]) / base[metric]

            results[size] = tax

        return results

    def fit_scaling_law(
        self,
        sizes: list[int],
        values: list[float],
        law_type: str = "power",
    ) -> dict[str, float]:
        """Fit a scaling law to data.

        Supports power law: y = a * x^b + c

        Args:
            sizes: Model sizes
            values: Metric values
            law_type: Type of scaling law

        Returns:
            Fitted coefficients
        """
        sizes_arr = np.array(sizes, dtype=float)
        values_arr = np.array(values, dtype=float)

        if law_type == "power":
            # Power law: y = a * x^b + c
            def power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
                return a * np.power(x, b) + c

            # Initial guess
            p0 = [1.0, 0.5, 0.0]

            try:
                popt, _ = optimize.curve_fit(
                    power_law,
                    sizes_arr,
                    values_arr,
                    p0=p0,
                    maxfev=10000,
                )
                return {"a": popt[0], "b": popt[1], "c": popt[2], "type": "power"}
            except Exception:
                # Fallback to log-linear fit
                log_sizes = np.log(sizes_arr)
                coeffs = np.polyfit(log_sizes, values_arr, 1)
                return {"slope": coeffs[0], "intercept": coeffs[1], "type": "log_linear"}

        elif law_type == "log_linear":
            log_sizes = np.log(sizes_arr)
            coeffs = np.polyfit(log_sizes, values_arr, 1)
            return {"slope": coeffs[0], "intercept": coeffs[1], "type": "log_linear"}

        else:
            raise ValueError(f"Unknown law type: {law_type}")

    def predict_at_scale(
        self,
        coefficients: dict[str, float],
        target_size: int,
        law_type: str = "power",
    ) -> float:
        """Predict metric value at a target size.

        Args:
            coefficients: Scaling law coefficients
            target_size: Target model size
            law_type: Type of scaling law

        Returns:
            Predicted value
        """
        law_type = coefficients.get("type", law_type)

        if law_type == "power":
            a = coefficients["a"]
            b = coefficients["b"]
            c = coefficients["c"]
            return a * (target_size ** b) + c

        elif law_type == "log_linear":
            slope = coefficients["slope"]
            intercept = coefficients["intercept"]
            return slope * np.log(target_size) + intercept

        else:
            raise ValueError(f"Unknown law type: {law_type}")

    def detect_emergent_behaviors(
        self,
        models: dict[int, PreTrainedModel],
        eval_dataset: Dataset,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Detect emergent capabilities.

        Emergent behaviors appear suddenly at certain scales
        rather than improving gradually.

        Args:
            models: Models by size
            eval_dataset: Evaluation dataset
            threshold: Emergence threshold

        Returns:
            Detected emergent behaviors
        """
        emergent = []
        sizes = sorted(models.keys())

        if len(sizes) < 3:
            return emergent

        # Compute metrics
        prompt_column = "prompt" if "prompt" in eval_dataset.column_names else eval_dataset.column_names[0]
        prompts = eval_dataset[prompt_column][:30]

        metrics_by_size: dict[int, dict[str, float]] = {}
        for size, model in models.items():
            responses = self._generate_responses(model, prompts)
            metrics_by_size[size] = {
                "completion_rate": sum(1 for r in responses if len(r.strip()) > 10) / len(responses),
                "avg_length": np.mean([len(r.split()) for r in responses]),
                "coherence": self._compute_coherence(responses),
            }

        # Detect emergence (sudden jumps)
        for metric in ["completion_rate", "coherence"]:
            values = [metrics_by_size[s][metric] for s in sizes]

            for i in range(1, len(values)):
                prev_val = values[i - 1]
                curr_val = values[i]

                # Check for sudden improvement
                if prev_val > 0:
                    improvement = (curr_val - prev_val) / prev_val
                else:
                    improvement = curr_val

                if improvement > threshold:
                    emergent.append({
                        "behavior": metric,
                        "emerged_at_size": sizes[i],
                        "previous_size": sizes[i - 1],
                        "improvement": improvement,
                        "previous_value": prev_val,
                        "new_value": curr_val,
                    })

        return emergent

    def _add_analysis_notes(self, result: ScalingResult) -> None:
        """Add analysis observations to result."""
        # Check for power law fit quality
        for metric, coeffs in result.scaling_coefficients.items():
            if "b" in coeffs:
                b = coeffs["b"]
                if b > 0:
                    result.analysis_notes.append(
                        f"{metric}: Shows positive scaling (exponent={b:.3f})"
                    )
                elif b < -0.1:
                    result.analysis_notes.append(
                        f"{metric}: Shows diminishing returns (exponent={b:.3f})"
                    )

        # Check for potential emergent behaviors
        for metric, values in result.metrics.items():
            if len(values) >= 3:
                # Check for sudden jumps
                diffs = np.diff(values)
                if len(diffs) > 0:
                    max_jump_idx = np.argmax(np.abs(diffs))
                    max_jump = diffs[max_jump_idx]
                    avg_diff = np.mean(np.abs(diffs))

                    if avg_diff > 0 and abs(max_jump) > 2 * avg_diff:
                        result.analysis_notes.append(
                            f"{metric}: Potential emergence detected at size "
                            f"{result.model_sizes[max_jump_idx + 1]}"
                        )

    def generate_report(
        self,
        result: ScalingResult,
        output_path: str | Path,
    ) -> None:
        """Generate scaling analysis report.

        Args:
            result: Scaling results
            output_path: Output path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Scaling Analysis Report\n",
            f"Model Family: {self.model_family}\n",
            f"Sizes Analyzed: {', '.join(f'{s/1e6:.0f}M' for s in result.model_sizes)}\n",
            "\n## Metrics by Size\n",
        ]

        # Create table
        lines.append("| Size |" + " | ".join(result.metrics.keys()) + " |")
        lines.append("|------|" + " | ".join(["------"] * len(result.metrics)) + " |")

        for i, size in enumerate(result.model_sizes):
            row = [f"{size/1e6:.0f}M"]
            for metric, values in result.metrics.items():
                row.append(f"{values[i]:.4f}")
            lines.append("| " + " | ".join(row) + " |")

        # Scaling coefficients
        lines.append("\n## Scaling Law Fits\n")
        for metric, coeffs in result.scaling_coefficients.items():
            lines.append(f"### {metric}")
            if coeffs.get("type") == "power":
                lines.append(f"- Type: Power Law (y = a * x^b + c)")
                lines.append(f"- a: {coeffs['a']:.6f}")
                lines.append(f"- b: {coeffs['b']:.6f}")
                lines.append(f"- c: {coeffs['c']:.6f}")
            else:
                lines.append(f"- Type: Log-Linear")
                lines.append(f"- slope: {coeffs['slope']:.6f}")
                lines.append(f"- intercept: {coeffs['intercept']:.6f}")
            lines.append("")

        # Predictions
        if result.predictions:
            lines.append("\n## Predictions\n")
            for metric, preds in result.predictions.items():
                lines.append(f"### {metric}")
                for size, pred in preds.items():
                    lines.append(f"- {size/1e9:.1f}B: {pred:.4f}")
                lines.append("")

        # Analysis notes
        if result.analysis_notes:
            lines.append("\n## Analysis Notes\n")
            for note in result.analysis_notes:
                lines.append(f"- {note}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))


def plot_scaling_curves(
    result: ScalingResult,
    metrics: list[str] | None = None,
    log_scale: bool = True,
    show_fit: bool = True,
    save_path: str | None = None,
) -> Any:
    """Plot scaling curves for metrics.

    Args:
        result: Scaling analysis results
        metrics: Metrics to plot (None = all)
        log_scale: Use log scale for x-axis
        show_fit: Show fitted scaling law
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return None

    metrics = metrics or list(result.metrics.keys())
    n_metrics = len(metrics)

    if n_metrics == 0:
        return None

    # Determine grid
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    sizes = np.array(result.model_sizes)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = np.array(result.metrics[metric])

        # Plot data points
        if log_scale:
            ax.semilogx(sizes, values, "o-", markersize=8, linewidth=2, label="Data")
        else:
            ax.plot(sizes, values, "o-", markersize=8, linewidth=2, label="Data")

        # Plot fitted curve
        if show_fit and metric in result.scaling_coefficients:
            coeffs = result.scaling_coefficients[metric]
            x_fit = np.logspace(np.log10(sizes.min()), np.log10(sizes.max() * 10), 100)

            analyzer = ScalingAnalyzer()
            y_fit = [analyzer.predict_at_scale(coeffs, x) for x in x_fit]

            if log_scale:
                ax.semilogx(x_fit, y_fit, "--", alpha=0.7, label="Fit")
            else:
                ax.plot(x_fit, y_fit, "--", alpha=0.7, label="Fit")

        ax.set_xlabel("Model Size (parameters)")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} Scaling")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format x-axis
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.1f}B")
        )

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_alignment_tax(
    base_result: ScalingResult,
    aligned_result: ScalingResult,
    metrics: list[str] | None = None,
    save_path: str | None = None,
) -> Any:
    """Plot alignment tax across scales.

    Args:
        base_result: Results for base models
        aligned_result: Results for aligned models
        metrics: Metrics to compare
        save_path: Optional save path

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return None

    # Find common metrics
    common_metrics = set(base_result.metrics.keys()) & set(aligned_result.metrics.keys())
    metrics = metrics or list(common_metrics)

    if not metrics:
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    sizes = np.array(base_result.model_sizes)

    for i, metric in enumerate(metrics):
        ax = axes[i]

        base_values = np.array(base_result.metrics[metric])
        aligned_values = np.array(aligned_result.metrics[metric])

        # Plot both
        ax.semilogx(sizes, base_values, "o-", label="Base", markersize=8)
        ax.semilogx(sizes, aligned_values, "s-", label="Aligned", markersize=8)

        # Shade the gap
        ax.fill_between(sizes, base_values, aligned_values, alpha=0.3)

        ax.set_xlabel("Model Size")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric}: Alignment Tax")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_scaling_table(
    result: ScalingResult,
    metrics: list[str] | None = None,
) -> str:
    """Create markdown table of scaling results.

    Args:
        result: Scaling results
        metrics: Metrics to include

    Returns:
        Markdown table string
    """
    metrics = metrics or list(result.metrics.keys())

    lines = []

    # Header
    header = "| Size |" + " | ".join(metrics) + " |"
    lines.append(header)

    # Separator
    sep = "|------|" + " | ".join(["------" for _ in metrics]) + " |"
    lines.append(sep)

    # Data rows
    for i, size in enumerate(result.model_sizes):
        row = [f"{size/1e6:.0f}M"]
        for metric in metrics:
            if metric in result.metrics and i < len(result.metrics[metric]):
                row.append(f"{result.metrics[metric][i]:.4f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# Convenience function
def run_scaling_experiment(
    model_sizes: list[int] | None = None,
    model_family: str = "pythia",
    eval_prompts: list[str] | None = None,
    reward_model: RewardModel | None = None,
) -> ScalingResult:
    """Run a complete scaling experiment.

    Args:
        model_sizes: Model sizes to analyze
        model_family: Model family
        eval_prompts: Evaluation prompts
        reward_model: Optional reward model

    Returns:
        Scaling results
    """
    model_sizes = model_sizes or [70_000_000, 410_000_000, 1_000_000_000]

    # Default prompts
    if eval_prompts is None:
        eval_prompts = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a haiku about coding.",
            "What are the benefits of exercise?",
            "How does gravity work?",
        ]

    # Create dataset
    eval_dataset = Dataset.from_dict({"prompt": eval_prompts})

    # Get model paths
    if model_family.lower() == "pythia":
        model_paths = {
            size: PYTHIA_SIZES.get(size, f"EleutherAI/pythia-{size//1_000_000}m")
            for size in model_sizes
            if size in PYTHIA_SIZES
        }
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(list(model_paths.values())[0])

    # Run analysis
    analyzer = ScalingAnalyzer(
        model_family=model_family,
        model_sizes=model_sizes,
        reward_model=reward_model,
        tokenizer=tokenizer,
    )

    return analyzer.analyze_from_paths(model_paths, eval_dataset)
