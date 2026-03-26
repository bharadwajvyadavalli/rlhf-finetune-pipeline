#!/usr/bin/env python3
"""
Run PPO vs DPO comparison evaluation.

This script compares PPO and DPO aligned models on various metrics
to understand the trade-offs between approaches.

Usage:
    python scripts/run_comparison.py --ppo-checkpoint ./outputs/ppo --dpo-checkpoint ./outputs/dpo
    python scripts/run_comparison.py --ppo-checkpoint ./outputs/ppo --dpo-checkpoint ./outputs/dpo --reward-model ./outputs/reward
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.comparison import AlignmentComparator, ComparisonResult
from evaluation.benchmarks import MTBenchEvaluator, AlpacaEvalRunner
from evaluation.reward_hacking import RewardHackingDetector
from reward.reward_model import RewardModel


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_level: Logging level
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run PPO vs DPO Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        required=True,
        help="Path to PPO model checkpoint",
    )
    parser.add_argument(
        "--dpo-checkpoint",
        type=str,
        required=True,
        help="Path to DPO model checkpoint",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=None,
        help="Path to reward model (for reward-based evaluation)",
    )
    parser.add_argument(
        "--ref-model",
        type=str,
        default=None,
        help="Path to reference model (for KL computation)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default="Anthropic/hh-rlhf",
        help="Evaluation dataset",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation",
    )

    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )

    # Benchmark arguments
    parser.add_argument(
        "--run-mt-bench",
        action="store_true",
        help="Run MT-Bench evaluation",
    )
    parser.add_argument(
        "--run-alpaca-eval",
        action="store_true",
        help="Run AlpacaEval evaluation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4",
        help="Judge model for benchmarks",
    )

    # Analysis arguments
    parser.add_argument(
        "--detect-hacking",
        action="store_true",
        help="Run reward hacking detection",
    )
    parser.add_argument(
        "--compare-lengths",
        action="store_true",
        help="Compare response lengths",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save sample comparisons",
    )
    parser.add_argument(
        "--num-sample-comparisons",
        type=int,
        default=20,
        help="Number of sample comparisons to save",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def load_models(args: argparse.Namespace) -> dict[str, Any]:
    """Load models for comparison.

    Args:
        args: Parsed arguments

    Returns:
        Dictionary of loaded models
    """
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}

    # Load PPO model
    logger.info(f"Loading PPO model from {args.ppo_checkpoint}")
    models["ppo"] = AutoModelForCausalLM.from_pretrained(
        args.ppo_checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Load DPO model
    logger.info(f"Loading DPO model from {args.dpo_checkpoint}")
    models["dpo"] = AutoModelForCausalLM.from_pretrained(
        args.dpo_checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Load tokenizer (from PPO checkpoint)
    models["tokenizer"] = AutoTokenizer.from_pretrained(args.ppo_checkpoint)
    if models["tokenizer"].pad_token is None:
        models["tokenizer"].pad_token = models["tokenizer"].eos_token

    # Load reward model if provided
    if args.reward_model:
        logger.info(f"Loading reward model from {args.reward_model}")
        models["reward"] = RewardModel.from_pretrained(args.reward_model)
        models["reward"] = models["reward"].to(device)
        models["reward"].eval()

    # Load reference model if provided
    if args.ref_model:
        logger.info(f"Loading reference model from {args.ref_model}")
        models["ref"] = AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        models["ref"].eval()

    return models


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Run the comparison evaluation.

    Args:
        args: Parsed arguments

    Returns:
        Comparison results
    """
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    # Load models
    models = load_models(args)
    tokenizer = models["tokenizer"]
    ppo_model = models["ppo"]
    dpo_model = models["dpo"]
    reward_model = models.get("reward")
    ref_model = models.get("ref")

    # Load evaluation dataset
    logger.info(f"Loading evaluation dataset {args.eval_dataset}")
    try:
        dataset = load_dataset(args.eval_dataset)
        if "test" in dataset:
            eval_data = dataset["test"]
        elif "validation" in dataset:
            eval_data = dataset["validation"]
        else:
            eval_data = dataset["train"]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Extract prompts
    def extract_prompt(example: dict[str, Any]) -> str:
        if "prompt" in example:
            return example["prompt"]
        elif "question" in example:
            return example["question"]
        elif "instruction" in example:
            return example["instruction"]
        elif "chosen" in example:
            chosen = example["chosen"]
            if isinstance(chosen, str) and "Human:" in chosen:
                parts = chosen.split("Human:")
                if len(parts) > 1:
                    return "Human:" + parts[1].split("Assistant:")[0].strip()
            return str(chosen)[:200]
        else:
            keys = list(example.keys())
            return str(example.get(keys[0], ""))[:200]

    prompts = [extract_prompt(ex) for ex in eval_data][:args.num_samples]
    logger.info(f"Extracted {len(prompts)} prompts for evaluation")

    results = {
        "timestamp": datetime.now().isoformat(),
        "ppo_checkpoint": args.ppo_checkpoint,
        "dpo_checkpoint": args.dpo_checkpoint,
        "num_samples": len(prompts),
        "metrics": {},
        "samples": [],
    }

    # Generate responses from both models
    logger.info("Generating responses from PPO model...")
    ppo_responses = generate_responses(ppo_model, tokenizer, prompts, args)

    logger.info("Generating responses from DPO model...")
    dpo_responses = generate_responses(dpo_model, tokenizer, prompts, args)

    # Compute basic metrics
    logger.info("Computing basic metrics...")
    results["metrics"]["ppo"] = compute_basic_metrics(ppo_responses)
    results["metrics"]["dpo"] = compute_basic_metrics(dpo_responses)

    # Compute reward metrics if reward model available
    if reward_model:
        logger.info("Computing reward metrics...")
        ppo_rewards = compute_rewards(reward_model, prompts, ppo_responses)
        dpo_rewards = compute_rewards(reward_model, prompts, dpo_responses)

        results["metrics"]["ppo"]["mean_reward"] = float(ppo_rewards.mean())
        results["metrics"]["ppo"]["std_reward"] = float(ppo_rewards.std())
        results["metrics"]["dpo"]["mean_reward"] = float(dpo_rewards.mean())
        results["metrics"]["dpo"]["std_reward"] = float(dpo_rewards.std())

        # Win rate based on reward
        ppo_wins = (ppo_rewards > dpo_rewards).sum().item()
        dpo_wins = (dpo_rewards > ppo_rewards).sum().item()
        ties = (ppo_rewards == dpo_rewards).sum().item()

        results["metrics"]["reward_win_rates"] = {
            "ppo_wins": ppo_wins / len(prompts),
            "dpo_wins": dpo_wins / len(prompts),
            "ties": ties / len(prompts),
        }

    # Compute KL divergence if reference model available
    if ref_model:
        logger.info("Computing KL divergence...")
        ppo_kl = compute_kl_divergence(ppo_model, ref_model, tokenizer, prompts[:50])
        dpo_kl = compute_kl_divergence(dpo_model, ref_model, tokenizer, prompts[:50])

        results["metrics"]["ppo"]["kl_divergence"] = ppo_kl
        results["metrics"]["dpo"]["kl_divergence"] = dpo_kl

    # Run reward hacking detection
    if args.detect_hacking and reward_model:
        logger.info("Running reward hacking detection...")
        detector = RewardHackingDetector(reward_model=reward_model, tokenizer=tokenizer)

        ppo_hacking = detector.analyze_responses(prompts[:100], ppo_responses[:100])
        dpo_hacking = detector.analyze_responses(prompts[:100], dpo_responses[:100])

        results["metrics"]["ppo"]["hacking_risk"] = ppo_hacking.overall_risk
        results["metrics"]["dpo"]["hacking_risk"] = dpo_hacking.overall_risk
        results["hacking_analysis"] = {
            "ppo": ppo_hacking.to_dict(),
            "dpo": dpo_hacking.to_dict(),
        }

    # Run MT-Bench if requested
    if args.run_mt_bench:
        logger.info("Running MT-Bench evaluation...")
        mt_bench = MTBenchEvaluator(judge_model=args.judge_model, tokenizer=tokenizer)

        ppo_mt_result = mt_bench.evaluate(model=ppo_model, model_name="PPO")
        dpo_mt_result = mt_bench.evaluate(model=dpo_model, model_name="DPO")

        results["metrics"]["ppo"]["mt_bench_score"] = ppo_mt_result.overall_score
        results["metrics"]["dpo"]["mt_bench_score"] = dpo_mt_result.overall_score
        results["mt_bench"] = {
            "ppo": ppo_mt_result.to_dict(),
            "dpo": dpo_mt_result.to_dict(),
        }

    # Run AlpacaEval if requested
    if args.run_alpaca_eval:
        logger.info("Running AlpacaEval...")
        alpaca_eval = AlpacaEvalRunner(tokenizer=tokenizer)

        ppo_alpaca_result = alpaca_eval.evaluate(model=ppo_model, model_name="PPO")
        dpo_alpaca_result = alpaca_eval.evaluate(model=dpo_model, model_name="DPO")

        results["metrics"]["ppo"]["alpaca_eval_score"] = ppo_alpaca_result.overall_score
        results["metrics"]["dpo"]["alpaca_eval_score"] = dpo_alpaca_result.overall_score

    # Save sample comparisons
    if args.save_samples:
        logger.info("Saving sample comparisons...")
        for i in range(min(args.num_sample_comparisons, len(prompts))):
            results["samples"].append({
                "prompt": prompts[i],
                "ppo_response": ppo_responses[i],
                "dpo_response": dpo_responses[i],
            })

    return results


def generate_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
) -> list[str]:
    """Generate responses from a model.

    Args:
        model: Model to generate from
        tokenizer: Tokenizer
        prompts: Input prompts
        args: Arguments

    Returns:
        Generated responses
    """
    device = next(model.parameters()).device
    model.eval()
    responses = []

    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + args.batch_size]

        # Format prompts
        formatted = []
        for prompt in batch_prompts:
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = f"User: {prompt}\nAssistant:"
            formatted.append(text)

        # Tokenize
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Decode
        for j, output in enumerate(outputs):
            response = tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            responses.append(response)

    return responses


def compute_basic_metrics(responses: list[str]) -> dict[str, float]:
    """Compute basic response metrics.

    Args:
        responses: Model responses

    Returns:
        Metrics dictionary
    """
    import numpy as np

    lengths = [len(r.split()) for r in responses]

    return {
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": float(np.min(lengths)),
        "max_length": float(np.max(lengths)),
        "empty_rate": sum(1 for r in responses if len(r.strip()) == 0) / len(responses),
    }


def compute_rewards(
    reward_model: Any,
    prompts: list[str],
    responses: list[str],
) -> torch.Tensor:
    """Compute rewards for responses.

    Args:
        reward_model: Reward model
        prompts: Prompts
        responses: Responses

    Returns:
        Reward tensor
    """
    texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
    return reward_model.compute_rewards(texts)


def compute_kl_divergence(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    prompts: list[str],
) -> float:
    """Compute KL divergence from reference.

    Args:
        model: Policy model
        ref_model: Reference model
        tokenizer: Tokenizer
        prompts: Prompts

    Returns:
        Average KL divergence
    """
    import torch.nn.functional as F

    device = next(model.parameters()).device
    model.eval()
    ref_model.eval()

    kl_divs = []

    for prompt in prompts:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"User: {prompt}\nAssistant:"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            model_logits = model(**inputs).logits
            ref_logits = ref_model(**inputs).logits

            # Compute KL divergence
            model_probs = F.softmax(model_logits, dim=-1)
            ref_probs = F.softmax(ref_logits, dim=-1)

            kl = F.kl_div(model_probs.log(), ref_probs, reduction="batchmean")
            kl_divs.append(kl.item())

    return float(sum(kl_divs) / len(kl_divs))


def generate_report(
    results: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Generate comparison report.

    Args:
        results: Comparison results
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate markdown report
    lines = [
        "# PPO vs DPO Comparison Report\n",
        f"Generated: {results['timestamp']}\n",
        "",
        "## Models",
        f"- PPO: `{results['ppo_checkpoint']}`",
        f"- DPO: `{results['dpo_checkpoint']}`",
        f"- Samples evaluated: {results['num_samples']}",
        "",
        "## Summary Metrics\n",
        "| Metric | PPO | DPO |",
        "|--------|-----|-----|",
    ]

    ppo_metrics = results["metrics"].get("ppo", {})
    dpo_metrics = results["metrics"].get("dpo", {})

    for metric in set(ppo_metrics.keys()) | set(dpo_metrics.keys()):
        ppo_val = ppo_metrics.get(metric, "N/A")
        dpo_val = dpo_metrics.get(metric, "N/A")
        if isinstance(ppo_val, float):
            ppo_val = f"{ppo_val:.4f}"
        if isinstance(dpo_val, float):
            dpo_val = f"{dpo_val:.4f}"
        lines.append(f"| {metric} | {ppo_val} | {dpo_val} |")

    # Win rates
    if "reward_win_rates" in results["metrics"]:
        win_rates = results["metrics"]["reward_win_rates"]
        lines.extend([
            "",
            "## Reward-Based Win Rates",
            f"- PPO wins: {win_rates['ppo_wins']:.1%}",
            f"- DPO wins: {win_rates['dpo_wins']:.1%}",
            f"- Ties: {win_rates['ties']:.1%}",
        ])

    # Hacking analysis
    if "hacking_analysis" in results:
        lines.extend([
            "",
            "## Reward Hacking Analysis",
            f"- PPO hacking risk: {results['metrics']['ppo'].get('hacking_risk', 'N/A')}",
            f"- DPO hacking risk: {results['metrics']['dpo'].get('hacking_risk', 'N/A')}",
        ])

    # Sample comparisons
    if results.get("samples"):
        lines.extend([
            "",
            "## Sample Comparisons",
        ])
        for i, sample in enumerate(results["samples"][:5]):
            lines.extend([
                f"\n### Sample {i + 1}",
                f"**Prompt:** {sample['prompt'][:200]}...",
                "",
                "**PPO Response:**",
                f"```\n{sample['ppo_response'][:500]}...\n```",
                "",
                "**DPO Response:**",
                f"```\n{sample['dpo_response'][:500]}...\n```",
            ])

    # Write report
    with open(output_dir / "report.md", "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting PPO vs DPO comparison")
    logger.info(f"PPO checkpoint: {args.ppo_checkpoint}")
    logger.info(f"DPO checkpoint: {args.dpo_checkpoint}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run comparison
    try:
        results = run_comparison(args)
        generate_report(results, output_dir)
        logger.info(f"Comparison completed. Results saved to {output_dir}")
        return 0
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
