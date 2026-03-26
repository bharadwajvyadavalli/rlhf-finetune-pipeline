"""
Head-to-head PPO vs DPO comparison framework.

This module provides tools for comparing PPO and DPO alignment approaches
on the same base model and data. Features:

- Standardized evaluation metrics
- Win rate computation
- Statistical significance testing
- Visualization of results

Understanding the trade-offs between PPO and DPO is crucial for
choosing the right alignment approach.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    )
except ImportError:
    PreTrainedModel = None
    PreTrainedTokenizer = None

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from reward.reward_model import RewardModel


@dataclass
class ComparisonResult:
    """Result of comparing two aligned models.

    Attributes:
        model_a_name: Name of first model
        model_b_name: Name of second model
        metrics: Dictionary of comparison metrics
        win_rates: Win rates for each model
        samples: Sample comparisons
        statistical_tests: Statistical test results
    """

    model_a_name: str = "PPO"
    model_b_name: str = "DPO"
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    win_rates: dict[str, float] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    statistical_tests: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "metrics": self.metrics,
            "win_rates": self.win_rates,
            "samples": self.samples,
            "statistical_tests": self.statistical_tests,
        }

    def summary(self) -> str:
        """Get a text summary of results.

        Returns:
            Summary string
        """
        lines = []
        lines.append(f"Comparison: {self.model_a_name} vs {self.model_b_name}")
        lines.append("=" * 50)

        # Win rates
        if self.win_rates:
            lines.append("\nWin Rates:")
            for key, value in self.win_rates.items():
                lines.append(f"  {key}: {value:.2%}")

        # Metrics
        if self.metrics:
            lines.append("\nMetrics:")
            for model_name, model_metrics in self.metrics.items():
                lines.append(f"\n  {model_name}:")
                for metric, value in model_metrics.items():
                    lines.append(f"    {metric}: {value:.4f}")

        # Statistical tests
        if self.statistical_tests:
            lines.append("\nStatistical Tests:")
            for test_name, results in self.statistical_tests.items():
                lines.append(f"  {test_name}:")
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, float):
                            lines.append(f"    {key}: {value:.4f}")
                        else:
                            lines.append(f"    {key}: {value}")

        return "\n".join(lines)


class AlignmentComparator:
    """Comparator for alignment approaches.

    Provides tools for head-to-head comparison of aligned models
    using various metrics and evaluation methods.

    Args:
        reward_model: Reward model for scoring
        tokenizer: Tokenizer for text processing
        judge_model: Optional judge model for pairwise comparison
    """

    def __init__(
        self,
        reward_model: RewardModel | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        judge_model: PreTrainedModel | None = None,
    ) -> None:
        """Initialize the comparator.

        Args:
            reward_model: Reward model
            tokenizer: Tokenizer
            judge_model: Judge model for pairwise eval
        """
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.judge_model = judge_model

    def compare(
        self,
        model_a: PreTrainedModel,
        model_b: PreTrainedModel,
        eval_dataset: Dataset,
        model_a_name: str = "PPO",
        model_b_name: str = "DPO",
    ) -> ComparisonResult:
        """Compare two aligned models.

        Args:
            model_a: First model
            model_b: Second model
            eval_dataset: Evaluation dataset
            model_a_name: Name for first model
            model_b_name: Name for second model

        Returns:
            Comparison results
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_a.to(device)
        model_b.to(device)
        model_a.eval()
        model_b.eval()

        # Get prompts from dataset
        if hasattr(eval_dataset, "__getitem__"):
            prompts = [eval_dataset[i]["prompt"] for i in range(len(eval_dataset))]
        else:
            prompts = list(eval_dataset["prompt"])

        # Generate responses
        responses_a = self.generate_responses(model_a, prompts)
        responses_b = self.generate_responses(model_b, prompts)

        # Compute reward-based metrics
        reward_metrics = self.compute_reward_metrics(prompts, responses_a, responses_b)

        # Compute win rates
        if self.reward_model is not None:
            rewards_a = self.reward_model.compute_rewards(prompts, responses_a)
            rewards_b = self.reward_model.compute_rewards(prompts, responses_b)
            win_rates = self.compute_win_rates(rewards_a, rewards_b)
        else:
            rewards_a = torch.zeros(len(prompts))
            rewards_b = torch.zeros(len(prompts))
            win_rates = {model_a_name: 0.5, model_b_name: 0.5, "ties": 0.0}

        # Compute length statistics
        length_stats = self.compute_length_statistics(responses_a, responses_b)

        # Run statistical tests
        statistical_tests = {}
        if SCIPY_AVAILABLE and self.reward_model is not None:
            statistical_tests = self.run_statistical_tests(rewards_a, rewards_b)

        # Sample comparisons
        samples = self.sample_comparisons(
            prompts, responses_a, responses_b, rewards_a, rewards_b
        )

        # Combine metrics
        metrics = {
            model_a_name: {
                **reward_metrics.get(model_a_name, {}),
                **length_stats.get(model_a_name, {}),
            },
            model_b_name: {
                **reward_metrics.get(model_b_name, {}),
                **length_stats.get(model_b_name, {}),
            },
        }

        # Update win rates with model names
        named_win_rates = {
            model_a_name: win_rates.get("model_a", win_rates.get(model_a_name, 0)),
            model_b_name: win_rates.get("model_b", win_rates.get(model_b_name, 0)),
            "ties": win_rates.get("ties", 0),
        }

        return ComparisonResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            metrics=metrics,
            win_rates=named_win_rates,
            samples=samples,
            statistical_tests=statistical_tests,
        )

    def generate_responses(
        self,
        model: PreTrainedModel,
        prompts: list[str],
        **generation_kwargs: Any,
    ) -> list[str]:
        """Generate responses from a model.

        Args:
            model: Language model
            prompts: Input prompts
            **generation_kwargs: Generation parameters

        Returns:
            Generated responses
        """
        device = next(model.parameters()).device
        responses = []

        # Default generation params
        default_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }
        default_kwargs.update(generation_kwargs)

        for prompt in tqdm(prompts, desc="Generating responses"):
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **default_kwargs,
                )

            response = self.tokenizer.decode(
                outputs[0][encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)

        return responses

    def compute_reward_metrics(
        self,
        prompts: list[str],
        responses_a: list[str],
        responses_b: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute reward-based metrics.

        Args:
            prompts: Input prompts
            responses_a: Responses from model A
            responses_b: Responses from model B

        Returns:
            Metrics for each model
        """
        if self.reward_model is None:
            return {}

        rewards_a = self.reward_model.compute_rewards(prompts, responses_a)
        rewards_b = self.reward_model.compute_rewards(prompts, responses_b)

        return {
            "model_a": {
                "reward_mean": rewards_a.mean().item(),
                "reward_std": rewards_a.std().item(),
                "reward_min": rewards_a.min().item(),
                "reward_max": rewards_a.max().item(),
            },
            "model_b": {
                "reward_mean": rewards_b.mean().item(),
                "reward_std": rewards_b.std().item(),
                "reward_min": rewards_b.min().item(),
                "reward_max": rewards_b.max().item(),
            },
        }

    def compute_win_rates(
        self,
        rewards_a: torch.Tensor,
        rewards_b: torch.Tensor,
    ) -> dict[str, float]:
        """Compute win rates between models.

        Args:
            rewards_a: Rewards for model A
            rewards_b: Rewards for model B

        Returns:
            Win rates dictionary
        """
        a_wins = (rewards_a > rewards_b).float().sum().item()
        b_wins = (rewards_b > rewards_a).float().sum().item()
        ties = (rewards_a == rewards_b).float().sum().item()
        total = len(rewards_a)

        return {
            "model_a": a_wins / total,
            "model_b": b_wins / total,
            "ties": ties / total,
        }

    def compute_pairwise_win_rate(
        self,
        prompts: list[str],
        responses_a: list[str],
        responses_b: list[str],
    ) -> dict[str, float]:
        """Compute pairwise win rate using judge model.

        Args:
            prompts: Input prompts
            responses_a: Responses from model A
            responses_b: Responses from model B

        Returns:
            Pairwise win rates
        """
        if self.judge_model is None:
            return {"model_a": 0.5, "model_b": 0.5, "ties": 0.0}

        # Use judge model for pairwise comparison
        device = next(self.judge_model.parameters()).device
        a_wins = 0
        b_wins = 0
        ties = 0

        judge_prompt_template = """Compare these two responses and determine which is better.

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response is better? Answer with just "A", "B", or "Tie":"""

        for prompt, resp_a, resp_b in tqdm(
            zip(prompts, responses_a, responses_b),
            desc="Judging responses",
            total=len(prompts),
        ):
            judge_prompt = judge_prompt_template.format(
                prompt=prompt,
                response_a=resp_a,
                response_b=resp_b,
            )

            encoded = self.tokenizer(
                judge_prompt,
                return_tensors="pt",
                truncation=True,
            ).to(device)

            with torch.no_grad():
                outputs = self.judge_model.generate(
                    **encoded,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            verdict = self.tokenizer.decode(
                outputs[0][encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip().upper()

            if "A" in verdict and "B" not in verdict:
                a_wins += 1
            elif "B" in verdict and "A" not in verdict:
                b_wins += 1
            else:
                ties += 1

        total = len(prompts)
        return {
            "model_a": a_wins / total,
            "model_b": b_wins / total,
            "ties": ties / total,
        }

    def compute_length_statistics(
        self,
        responses_a: list[str],
        responses_b: list[str],
    ) -> dict[str, dict[str, float]]:
        """Compute response length statistics.

        Args:
            responses_a: Responses from model A
            responses_b: Responses from model B

        Returns:
            Length statistics
        """
        lengths_a = [len(r) for r in responses_a]
        lengths_b = [len(r) for r in responses_b]

        return {
            "model_a": {
                "length_mean": np.mean(lengths_a),
                "length_std": np.std(lengths_a),
                "length_min": np.min(lengths_a),
                "length_max": np.max(lengths_a),
            },
            "model_b": {
                "length_mean": np.mean(lengths_b),
                "length_std": np.std(lengths_b),
                "length_min": np.min(lengths_b),
                "length_max": np.max(lengths_b),
            },
        }

    def compute_kl_divergence(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        prompts: list[str],
        responses: list[str],
    ) -> float:
        """Compute KL divergence from reference model.

        Args:
            model: Aligned model
            ref_model: Reference model
            prompts: Input prompts
            responses: Generated responses

        Returns:
            Mean KL divergence
        """
        device = next(model.parameters()).device
        total_kl = 0.0
        count = 0

        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device)

            with torch.no_grad():
                model_outputs = model(**encoded)
                ref_outputs = ref_model(**encoded)

                model_logprobs = torch.log_softmax(model_outputs.logits, dim=-1)
                ref_logprobs = torch.log_softmax(ref_outputs.logits, dim=-1)

                kl = (torch.exp(model_logprobs) * (model_logprobs - ref_logprobs)).sum(dim=-1)
                total_kl += kl.mean().item()
                count += 1

        return total_kl / max(count, 1)

    def run_statistical_tests(
        self,
        rewards_a: torch.Tensor,
        rewards_b: torch.Tensor,
    ) -> dict[str, Any]:
        """Run statistical significance tests.

        Args:
            rewards_a: Rewards for model A
            rewards_b: Rewards for model B

        Returns:
            Test results (t-test, wilcoxon, etc.)
        """
        if not SCIPY_AVAILABLE:
            return {}

        rewards_a_np = rewards_a.detach().cpu().numpy()
        rewards_b_np = rewards_b.detach().cpu().numpy()

        results = {}

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(rewards_a_np, rewards_b_np)
        results["paired_t_test"] = {
            "statistic": float(t_stat),
            "p_value": float(t_pvalue),
            "significant_at_0.05": t_pvalue < 0.05,
        }

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pvalue = stats.wilcoxon(rewards_a_np, rewards_b_np)
            results["wilcoxon"] = {
                "statistic": float(w_stat),
                "p_value": float(w_pvalue),
                "significant_at_0.05": w_pvalue < 0.05,
            }
        except Exception:
            pass

        # Effect size (Cohen's d)
        diff = rewards_a_np - rewards_b_np
        cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
        results["effect_size"] = {
            "cohens_d": float(cohens_d),
            "interpretation": (
                "large" if abs(cohens_d) > 0.8 else
                "medium" if abs(cohens_d) > 0.5 else
                "small" if abs(cohens_d) > 0.2 else
                "negligible"
            ),
        }

        return results

    def sample_comparisons(
        self,
        prompts: list[str],
        responses_a: list[str],
        responses_b: list[str],
        rewards_a: torch.Tensor,
        rewards_b: torch.Tensor,
        num_samples: int = 10,
    ) -> list[dict[str, Any]]:
        """Sample representative comparisons.

        Args:
            prompts: Input prompts
            responses_a: Responses from A
            responses_b: Responses from B
            rewards_a: Rewards for A
            rewards_b: Rewards for B
            num_samples: Number of samples

        Returns:
            Sample comparison dictionaries
        """
        rewards_a_np = rewards_a.detach().cpu().numpy()
        rewards_b_np = rewards_b.detach().cpu().numpy()

        # Get indices where each model wins decisively
        diff = rewards_a_np - rewards_b_np
        sorted_indices = np.argsort(diff)

        # Sample from different regions
        samples = []
        n = len(prompts)

        # A wins strongly (top of sorted)
        for idx in sorted_indices[-num_samples // 3:]:
            samples.append({
                "prompt": prompts[idx],
                "response_a": responses_a[idx],
                "response_b": responses_b[idx],
                "reward_a": float(rewards_a_np[idx]),
                "reward_b": float(rewards_b_np[idx]),
                "winner": "A",
            })

        # B wins strongly (bottom of sorted)
        for idx in sorted_indices[:num_samples // 3]:
            samples.append({
                "prompt": prompts[idx],
                "response_a": responses_a[idx],
                "response_b": responses_b[idx],
                "reward_a": float(rewards_a_np[idx]),
                "reward_b": float(rewards_b_np[idx]),
                "winner": "B",
            })

        # Close comparisons (middle of sorted)
        mid_start = n // 2 - num_samples // 6
        mid_end = n // 2 + num_samples // 6
        for idx in sorted_indices[mid_start:mid_end]:
            samples.append({
                "prompt": prompts[idx],
                "response_a": responses_a[idx],
                "response_b": responses_b[idx],
                "reward_a": float(rewards_a_np[idx]),
                "reward_b": float(rewards_b_np[idx]),
                "winner": "Tie" if abs(diff[idx]) < 0.1 else ("A" if diff[idx] > 0 else "B"),
            })

        return samples[:num_samples]

    def plot_reward_distributions(
        self,
        rewards_a: torch.Tensor,
        rewards_b: torch.Tensor,
        names: tuple[str, str] = ("PPO", "DPO"),
        save_path: str | None = None,
    ) -> Any:
        """Plot reward distributions for both models.

        Args:
            rewards_a: Rewards for model A
            rewards_b: Rewards for model B
            names: Model names
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        fig, ax = plt.subplots(figsize=(10, 6))

        rewards_a_np = rewards_a.detach().cpu().numpy()
        rewards_b_np = rewards_b.detach().cpu().numpy()

        ax.hist(rewards_a_np, bins=30, alpha=0.6, label=names[0], color="blue")
        ax.hist(rewards_b_np, bins=30, alpha=0.6, label=names[1], color="orange")

        ax.axvline(rewards_a_np.mean(), color="blue", linestyle="--",
                   label=f"{names[0]} mean: {rewards_a_np.mean():.3f}")
        ax.axvline(rewards_b_np.mean(), color="orange", linestyle="--",
                   label=f"{names[1]} mean: {rewards_b_np.mean():.3f}")

        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.set_title("Reward Distribution Comparison")
        ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_length_vs_reward(
        self,
        responses_a: list[str],
        responses_b: list[str],
        rewards_a: torch.Tensor,
        rewards_b: torch.Tensor,
        names: tuple[str, str] = ("PPO", "DPO"),
        save_path: str | None = None,
    ) -> Any:
        """Plot length vs reward for both models.

        Args:
            responses_a: Responses from A
            responses_b: Responses from B
            rewards_a: Rewards for A
            rewards_b: Rewards for B
            names: Model names
            save_path: Optional save path

        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for plotting")

        fig, ax = plt.subplots(figsize=(10, 6))

        lengths_a = [len(r) for r in responses_a]
        lengths_b = [len(r) for r in responses_b]
        rewards_a_np = rewards_a.detach().cpu().numpy()
        rewards_b_np = rewards_b.detach().cpu().numpy()

        ax.scatter(lengths_a, rewards_a_np, alpha=0.5, label=names[0], s=20)
        ax.scatter(lengths_b, rewards_b_np, alpha=0.5, label=names[1], s=20)

        ax.set_xlabel("Response Length (characters)")
        ax.set_ylabel("Reward")
        ax.set_title("Length vs Reward Comparison")
        ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_report(
        self,
        result: ComparisonResult,
        output_path: str | Path,
    ) -> None:
        """Generate a comparison report.

        Args:
            result: Comparison results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Write markdown report
        report_lines = []
        report_lines.append(f"# Comparison: {result.model_a_name} vs {result.model_b_name}\n")

        # Win rates
        report_lines.append("## Win Rates\n")
        for model, rate in result.win_rates.items():
            report_lines.append(f"- {model}: {rate:.2%}")
        report_lines.append("")

        # Metrics
        report_lines.append("## Metrics\n")
        for model_name, metrics in result.metrics.items():
            report_lines.append(f"### {model_name}\n")
            for metric, value in metrics.items():
                report_lines.append(f"- {metric}: {value:.4f}")
            report_lines.append("")

        # Statistical tests
        if result.statistical_tests:
            report_lines.append("## Statistical Tests\n")
            for test_name, results in result.statistical_tests.items():
                report_lines.append(f"### {test_name}\n")
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, float):
                            report_lines.append(f"- {key}: {value:.4f}")
                        else:
                            report_lines.append(f"- {key}: {value}")
                report_lines.append("")

        # Write report
        with open(output_path / "comparison_report.md", "w") as f:
            f.write("\n".join(report_lines))

        # Save raw results
        with open(output_path / "comparison_results.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save samples
        if result.samples:
            with open(output_path / "sample_comparisons.json", "w") as f:
                json.dump(result.samples, f, indent=2)


def compare_models(
    ppo_model_path: str | Path,
    dpo_model_path: str | Path,
    eval_dataset: Dataset,
    reward_model_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> ComparisonResult:
    """Convenience function to compare PPO and DPO models.

    Args:
        ppo_model_path: Path to PPO model
        dpo_model_path: Path to DPO model
        eval_dataset: Evaluation dataset
        reward_model_path: Optional reward model path
        output_dir: Optional output directory

    Returns:
        Comparison results
    """
    # Load models
    ppo_model = AutoModelForCausalLM.from_pretrained(
        ppo_model_path, trust_remote_code=True, device_map="auto"
    )
    dpo_model = AutoModelForCausalLM.from_pretrained(
        dpo_model_path, trust_remote_code=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(ppo_model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load reward model if provided
    reward_model = None
    if reward_model_path is not None:
        reward_model = RewardModel.from_pretrained(reward_model_path)

    # Create comparator
    comparator = AlignmentComparator(
        reward_model=reward_model,
        tokenizer=tokenizer,
    )

    # Run comparison
    result = comparator.compare(
        ppo_model, dpo_model, eval_dataset,
        model_a_name="PPO", model_b_name="DPO"
    )

    # Generate report if output dir provided
    if output_dir is not None:
        comparator.generate_report(result, output_dir)

    return result


def load_comparison_result(
    result_path: str | Path,
) -> ComparisonResult:
    """Load a saved comparison result.

    Args:
        result_path: Path to saved result

    Returns:
        Loaded ComparisonResult
    """
    result_path = Path(result_path)

    with open(result_path / "comparison_results.json", "r") as f:
        data = json.load(f)

    return ComparisonResult(
        model_a_name=data["model_a_name"],
        model_b_name=data["model_b_name"],
        metrics=data["metrics"],
        win_rates=data["win_rates"],
        samples=data.get("samples", []),
        statistical_tests=data.get("statistical_tests", {}),
    )
