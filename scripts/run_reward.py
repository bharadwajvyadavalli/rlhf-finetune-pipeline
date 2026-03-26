#!/usr/bin/env python3
"""
Run Reward Model training.

This script trains a reward model using preference data for use
in PPO alignment or evaluation.

Usage:
    python scripts/run_reward.py --config configs/reward_config.yaml
    python scripts/run_reward.py --model meta-llama/Llama-2-7b-hf --dataset Anthropic/hh-rlhf
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from reward.reward_model import RewardModel, RewardModelConfig
from reward.reward_trainer import RewardTrainer, RewardTrainingConfig


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
        description="Run Reward Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reward_config.yaml",
        help="Path to configuration YAML file",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model name or path (overrides config)",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default=None,
        help="SFT checkpoint to initialize from (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Preference dataset name or path (overrides config)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (overrides config)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (overrides config)",
    )

    # Loss arguments
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["bradley_terry", "margin", "hinge"],
        default=None,
        help="Loss function type (overrides config)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help="Margin for margin-based loss (overrides config)",
    )

    # Other arguments
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=None,
        help="Number of layers to freeze (overrides config)",
    )
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
    if "training" not in config:
        config["training"] = {}
    if "loss" not in config:
        config["loss"] = {}

    # Model arguments
    if args.model:
        config["model"]["name"] = args.model
    if args.sft_checkpoint:
        config["model"]["sft_checkpoint"] = args.sft_checkpoint
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.freeze_layers is not None:
        config["model"]["freeze_layers"] = args.freeze_layers

    # Data arguments
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.max_seq_length:
        config["data"]["max_seq_length"] = args.max_seq_length

    # Training arguments
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["per_device_batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.gradient_accumulation_steps:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Loss arguments
    if args.loss_type:
        config["loss"]["type"] = args.loss_type
    if args.margin:
        config["loss"]["margin"] = args.margin

    # Other arguments
    if args.seed:
        config["training"]["seed"] = args.seed
    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume
    if args.wandb_project:
        config["training"]["wandb_project"] = args.wandb_project

    # Set defaults
    config["model"].setdefault("name", "EleutherAI/pythia-410m")
    config["model"].setdefault("freeze_layers", 0)
    config["data"].setdefault("dataset", "Anthropic/hh-rlhf")
    config["data"].setdefault("max_seq_length", 1024)
    config["training"].setdefault("output_dir", "outputs/reward")
    config["training"].setdefault("num_epochs", 1)
    config["training"].setdefault("per_device_batch_size", 4)
    config["training"].setdefault("learning_rate", 1e-5)
    config["training"].setdefault("gradient_accumulation_steps", 4)
    config["training"].setdefault("seed", 42)
    config["loss"].setdefault("type", "bradley_terry")
    config["loss"].setdefault("margin", 0.0)

    return config


def run_reward_training(config: dict[str, Any]) -> None:
    """Run reward model training with the given configuration.

    Args:
        config: Training configuration
    """
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Load tokenizer
    model_name = config["model"]["name"]
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create reward model config
    reward_config = RewardModelConfig(
        base_model_name=config["model"].get("sft_checkpoint", model_name),
        hidden_size=None,  # Auto-detect
        num_frozen_layers=config["model"]["freeze_layers"],
        reward_head_hidden_size=1024,
        reward_head_dropout=0.1,
    )

    # Load reward model
    logger.info(f"Loading reward model from {reward_config.base_model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_model = RewardModel(config=reward_config)
    reward_model = reward_model.to(device)
    logger.info(f"Reward model loaded on {device}")

    # Freeze layers if specified
    if config["model"]["freeze_layers"] > 0:
        reward_model.freeze_layers(config["model"]["freeze_layers"])
        logger.info(f"Froze first {config['model']['freeze_layers']} layers")

    # Load dataset
    dataset_name = config["data"]["dataset"]
    logger.info(f"Loading dataset {dataset_name}")

    try:
        dataset = load_dataset(dataset_name)
        if "train" in dataset:
            train_dataset = dataset["train"]
        else:
            train_dataset = dataset[list(dataset.keys())[0]]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # Process dataset to preference pairs format
    def process_preference_pair(example: dict[str, Any]) -> dict[str, Any]:
        # Handle different dataset formats
        if "chosen" in example and "rejected" in example:
            # Standard preference format
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Handle nested format (e.g., Anthropic/hh-rlhf)
            if isinstance(chosen, list):
                chosen = " ".join([m.get("content", "") for m in chosen])
            if isinstance(rejected, list):
                rejected = " ".join([m.get("content", "") for m in rejected])

            return {
                "prompt": example.get("prompt", ""),
                "chosen": chosen,
                "rejected": rejected,
            }
        elif "prompt" in example and "response_a" in example and "response_b" in example:
            # Alternative format
            if example.get("preference", 0) == 0:
                return {
                    "prompt": example["prompt"],
                    "chosen": example["response_a"],
                    "rejected": example["response_b"],
                }
            else:
                return {
                    "prompt": example["prompt"],
                    "chosen": example["response_b"],
                    "rejected": example["response_a"],
                }
        else:
            # Try to extract from available columns
            columns = list(example.keys())
            logger.warning(f"Unknown dataset format with columns: {columns}")
            return {
                "prompt": "",
                "chosen": str(example.get(columns[0], "")),
                "rejected": str(example.get(columns[1], "") if len(columns) > 1 else ""),
            }

    train_dataset = train_dataset.map(process_preference_pair)
    logger.info(f"Processed {len(train_dataset)} preference pairs")

    # Create training config
    training_config = RewardTrainingConfig(
        output_dir=config["training"]["output_dir"],
        num_epochs=config["training"]["num_epochs"],
        per_device_batch_size=config["training"]["per_device_batch_size"],
        learning_rate=config["training"]["learning_rate"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        max_seq_length=config["data"]["max_seq_length"],
        loss_type=config["loss"]["type"],
        margin=config["loss"]["margin"],
        seed=seed,
    )

    # Create trainer
    trainer = RewardTrainer(
        model=reward_model,
        tokenizer=tokenizer,
        config=training_config,
    )

    # Create preference pairs for training
    train_pairs = [
        (ex["prompt"], ex["chosen"], ex["rejected"])
        for ex in train_dataset
        if ex["chosen"] and ex["rejected"]
    ]
    logger.info(f"Training on {len(train_pairs)} valid preference pairs")

    # Train
    logger.info("Starting training...")
    trainer.train(train_pairs)

    # Save final model
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_model.save_pretrained(output_dir / "final")
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
    logger.info("Starting Reward Model training")

    # Load and merge config
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    logger.info(f"Configuration: {config}")

    # Run training
    try:
        run_reward_training(config)
        logger.info("Reward Model training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Reward Model training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
