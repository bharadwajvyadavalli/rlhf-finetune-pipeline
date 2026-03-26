#!/usr/bin/env python3
"""
Run PPO (Proximal Policy Optimization) alignment.

This script runs PPO training to align a language model using
a trained reward model.

Usage:
    python scripts/run_ppo.py --config configs/ppo_config.yaml
    python scripts/run_ppo.py --policy-model ./outputs/sft --reward-model ./outputs/reward
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from alignment.ppo_trainer import PPOTrainer, PPOConfig, GenerationConfig
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


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config or {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run PPO Alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_config.yaml",
        help="Path to configuration YAML file",
    )

    # Model arguments
    parser.add_argument(
        "--policy-model",
        type=str,
        default=None,
        help="Policy model path (overrides config)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=None,
        help="Reward model path (overrides config)",
    )
    parser.add_argument(
        "--ref-model",
        type=str,
        default=None,
        help="Reference model path (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    # Data arguments
    parser.add_argument(
        "--prompts-dataset",
        type=str,
        default=None,
        help="Dataset with prompts (overrides config)",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Maximum prompt length (overrides config)",
    )

    # PPO arguments
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Total training steps (overrides config)",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=None,
        help="Initial KL coefficient (overrides config)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Target KL divergence (overrides config)",
    )
    parser.add_argument(
        "--cliprange",
        type=float,
        default=None,
        help="PPO clip range (overrides config)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor (overrides config)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=None,
        help="GAE lambda (overrides config)",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rollout batch size (overrides config)",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=None,
        help="Mini-batch size for PPO updates (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=None,
        help="PPO epochs per batch (overrides config)",
    )

    # Generation arguments
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum new tokens to generate (overrides config)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (overrides config)",
    )

    # Other arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project name (overrides config)",
    )

    return parser.parse_args()


def merge_config_with_args(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Merge config file with command-line arguments.

    Args:
        config: Configuration dictionary
        args: Parsed arguments

    Returns:
        Merged configuration
    """
    # Create default structure if missing
    if "model" not in config:
        config["model"] = {}
    if "data" not in config:
        config["data"] = {}
    if "ppo" not in config:
        config["ppo"] = {}
    if "training" not in config:
        config["training"] = {}
    if "generation" not in config:
        config["generation"] = {}

    # Model arguments
    if args.policy_model:
        config["model"]["policy"] = args.policy_model
    if args.reward_model:
        config["model"]["reward"] = args.reward_model
    if args.ref_model:
        config["model"]["reference"] = args.ref_model
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    # Data arguments
    if args.prompts_dataset:
        config["data"]["prompts_dataset"] = args.prompts_dataset
    if args.max_prompt_length:
        config["data"]["max_prompt_length"] = args.max_prompt_length

    # PPO arguments
    if args.total_steps:
        config["training"]["total_steps"] = args.total_steps
    if args.kl_coef:
        config["ppo"]["kl_coef"] = args.kl_coef
    if args.target_kl:
        config["ppo"]["target_kl"] = args.target_kl
    if args.cliprange:
        config["ppo"]["cliprange"] = args.cliprange
    if args.gamma:
        config["ppo"]["gamma"] = args.gamma
    if args.lam:
        config["ppo"]["lam"] = args.lam

    # Training arguments
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.mini_batch_size:
        config["training"]["mini_batch_size"] = args.mini_batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.ppo_epochs:
        config["training"]["ppo_epochs"] = args.ppo_epochs

    # Generation arguments
    if args.max_new_tokens:
        config["generation"]["max_new_tokens"] = args.max_new_tokens
    if args.temperature:
        config["generation"]["temperature"] = args.temperature

    # Other arguments
    if args.seed:
        config["training"]["seed"] = args.seed
    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume
    if args.wandb_project:
        config["training"]["wandb_project"] = args.wandb_project

    # Set defaults
    config["model"].setdefault("policy", "outputs/sft/final")
    config["model"].setdefault("reward", "outputs/reward/final")
    config["data"].setdefault("prompts_dataset", "Anthropic/hh-rlhf")
    config["data"].setdefault("max_prompt_length", 512)
    config["training"].setdefault("output_dir", "outputs/ppo")
    config["training"].setdefault("total_steps", 1000)
    config["training"].setdefault("batch_size", 64)
    config["training"].setdefault("mini_batch_size", 8)
    config["training"].setdefault("learning_rate", 1e-6)
    config["training"].setdefault("ppo_epochs", 4)
    config["training"].setdefault("seed", 42)
    config["ppo"].setdefault("kl_coef", 0.1)
    config["ppo"].setdefault("target_kl", 6.0)
    config["ppo"].setdefault("cliprange", 0.2)
    config["ppo"].setdefault("gamma", 1.0)
    config["ppo"].setdefault("lam", 0.95)
    config["generation"].setdefault("max_new_tokens", 256)
    config["generation"].setdefault("temperature", 0.7)
    config["generation"].setdefault("top_p", 1.0)
    config["generation"].setdefault("do_sample", True)

    return config


def run_ppo(config: dict[str, Any]) -> None:
    """Run PPO training with the given configuration.

    Args:
        config: Training configuration
    """
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Load tokenizer
    policy_model_path = config["model"]["policy"]
    logger.info(f"Loading tokenizer from {policy_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(policy_model_path)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load policy model
    logger.info(f"Loading policy model from {policy_model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    logger.info(f"Policy model loaded on {device}")

    # Load reward model
    reward_model_path = config["model"]["reward"]
    logger.info(f"Loading reward model from {reward_model_path}")
    reward_model = RewardModel.from_pretrained(reward_model_path)
    reward_model = reward_model.to(device)
    reward_model.eval()
    logger.info("Reward model loaded")

    # Load prompts
    prompts_dataset = config["data"]["prompts_dataset"]
    logger.info(f"Loading prompts from {prompts_dataset}")

    try:
        dataset = load_dataset(prompts_dataset)
        if "train" in dataset:
            prompts_data = dataset["train"]
        else:
            prompts_data = dataset[list(dataset.keys())[0]]
    except Exception as e:
        logger.error(f"Failed to load prompts dataset: {e}")
        raise

    # Extract prompts
    def extract_prompt(example: dict[str, Any]) -> dict[str, str]:
        if "prompt" in example:
            return {"prompt": example["prompt"]}
        elif "question" in example:
            return {"prompt": example["question"]}
        elif "instruction" in example:
            return {"prompt": example["instruction"]}
        elif "chosen" in example:
            # Extract prompt from chosen response if it's a conversation
            chosen = example["chosen"]
            if isinstance(chosen, str) and "Human:" in chosen:
                # Extract human turn
                parts = chosen.split("Human:")
                if len(parts) > 1:
                    return {"prompt": "Human:" + parts[1].split("Assistant:")[0].strip()}
            return {"prompt": str(chosen)[:200]}
        else:
            keys = list(example.keys())
            return {"prompt": str(example.get(keys[0], ""))[:200]}

    prompts_data = prompts_data.map(extract_prompt)
    prompts = prompts_data["prompt"][:config["training"]["total_steps"]]
    logger.info(f"Loaded {len(prompts)} prompts")

    # Create PPO config
    ppo_config = PPOConfig(
        kl_coef=config["ppo"]["kl_coef"],
        target_kl=config["ppo"]["target_kl"],
        cliprange=config["ppo"]["cliprange"],
        cliprange_value=config["ppo"]["cliprange"],
        gamma=config["ppo"]["gamma"],
        lam=config["ppo"]["lam"],
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        mini_batch_size=config["training"]["mini_batch_size"],
        ppo_epochs=config["training"]["ppo_epochs"],
        seed=seed,
    )

    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=config["generation"]["max_new_tokens"],
        temperature=config["generation"]["temperature"],
        top_p=config["generation"]["top_p"],
        do_sample=config["generation"]["do_sample"],
    )

    # Create trainer
    trainer = PPOTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=ppo_config,
        generation_config=gen_config,
    )

    # Train
    logger.info("Starting PPO training...")
    total_steps = config["training"]["total_steps"]
    batch_size = config["training"]["batch_size"]
    num_batches = total_steps // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]

        if not batch_prompts:
            break

        stats = trainer.step(batch_prompts)
        logger.info(
            f"Batch {batch_idx + 1}/{num_batches}: "
            f"reward={stats.mean_reward:.4f}, "
            f"kl={stats.mean_kl:.4f}, "
            f"policy_loss={stats.policy_loss:.4f}"
        )

        # Save checkpoint periodically
        if (batch_idx + 1) % 10 == 0:
            output_dir = Path(config["training"]["output_dir"])
            checkpoint_dir = output_dir / f"checkpoint-{batch_idx + 1}"
            trainer.save(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Save final model
    output_dir = Path(config["training"]["output_dir"])
    trainer.save(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    logger.info(f"Saved final model to {output_dir / 'final'}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting PPO alignment")

    # Load and merge config
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    logger.info(f"Configuration: {config}")

    # Run training
    try:
        run_ppo(config)
        logger.info("PPO alignment completed successfully")
        return 0
    except Exception as e:
        logger.error(f"PPO alignment failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
