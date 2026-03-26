#!/usr/bin/env python3
"""
Run DPO (Direct Preference Optimization) alignment.

This script runs DPO training to align a language model directly
on preference data without an explicit reward model.

Usage:
    python scripts/run_dpo.py --config configs/dpo_config.yaml
    python scripts/run_dpo.py --model ./outputs/sft --dataset Anthropic/hh-rlhf
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

from alignment.dpo_trainer import DPOTrainer, DPOConfig, DPOLossType


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
        description="Run DPO Alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo_config.yaml",
        help="Path to configuration YAML file",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Policy model path (overrides config)",
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

    # DPO arguments
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="DPO beta parameter (overrides config)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        choices=["sigmoid", "hinge", "ipo", "kto"],
        default=None,
        help="DPO loss type (overrides config)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=None,
        help="Label smoothing factor (overrides config)",
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

    # Other arguments
    parser.add_argument(
        "--sync-ref-model",
        action="store_true",
        help="Sync reference model periodically",
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
    if "dpo" not in config:
        config["dpo"] = {}
    if "training" not in config:
        config["training"] = {}

    # Model arguments
    if args.model:
        config["model"]["policy"] = args.model
    if args.ref_model:
        config["model"]["reference"] = args.ref_model
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    # Data arguments
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.max_seq_length:
        config["data"]["max_seq_length"] = args.max_seq_length

    # DPO arguments
    if args.beta:
        config["dpo"]["beta"] = args.beta
    if args.loss_type:
        config["dpo"]["loss_type"] = args.loss_type
    if args.label_smoothing:
        config["dpo"]["label_smoothing"] = args.label_smoothing
    if args.sync_ref_model:
        config["dpo"]["sync_ref_model"] = True

    # Training arguments
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["per_device_batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.gradient_accumulation_steps:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    # Other arguments
    if args.seed:
        config["training"]["seed"] = args.seed
    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume
    if args.wandb_project:
        config["training"]["wandb_project"] = args.wandb_project

    # Set defaults
    config["model"].setdefault("policy", "outputs/sft/final")
    config["data"].setdefault("dataset", "Anthropic/hh-rlhf")
    config["data"].setdefault("max_seq_length", 1024)
    config["training"].setdefault("output_dir", "outputs/dpo")
    config["training"].setdefault("num_epochs", 1)
    config["training"].setdefault("per_device_batch_size", 4)
    config["training"].setdefault("learning_rate", 5e-7)
    config["training"].setdefault("gradient_accumulation_steps", 4)
    config["training"].setdefault("seed", 42)
    config["dpo"].setdefault("beta", 0.1)
    config["dpo"].setdefault("loss_type", "sigmoid")
    config["dpo"].setdefault("label_smoothing", 0.0)
    config["dpo"].setdefault("sync_ref_model", False)

    return config


def run_dpo(config: dict[str, Any]) -> None:
    """Run DPO training with the given configuration.

    Args:
        config: Training configuration
    """
    logger = logging.getLogger(__name__)

    # Set seed for reproducibility
    seed = config["training"].get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Load tokenizer
    model_path = config["model"]["policy"]
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load policy model
    logger.info(f"Loading policy model from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    logger.info(f"Policy model loaded on {device}")

    # Load reference model
    ref_model_path = config["model"].get("reference", model_path)
    logger.info(f"Loading reference model from {ref_model_path}")
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("Reference model loaded and frozen")

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
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Handle nested format
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
            columns = list(example.keys())
            return {
                "prompt": "",
                "chosen": str(example.get(columns[0], "")),
                "rejected": str(example.get(columns[1], "") if len(columns) > 1 else ""),
            }

    train_dataset = train_dataset.map(process_preference_pair)
    logger.info(f"Processed {len(train_dataset)} preference pairs")

    # Create DPO config
    loss_type_map = {
        "sigmoid": DPOLossType.SIGMOID,
        "hinge": DPOLossType.HINGE,
        "ipo": DPOLossType.IPO,
        "kto": DPOLossType.KTO,
    }
    loss_type = loss_type_map.get(config["dpo"]["loss_type"], DPOLossType.SIGMOID)

    dpo_config = DPOConfig(
        beta=config["dpo"]["beta"],
        loss_type=loss_type,
        label_smoothing=config["dpo"]["label_smoothing"],
        learning_rate=config["training"]["learning_rate"],
        num_epochs=config["training"]["num_epochs"],
        per_device_batch_size=config["training"]["per_device_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        max_seq_length=config["data"]["max_seq_length"],
        seed=seed,
    )

    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        config=dpo_config,
    )

    # Create preference pairs for training
    train_pairs = [
        (ex["prompt"], ex["chosen"], ex["rejected"])
        for ex in train_dataset
        if ex["chosen"] and ex["rejected"]
    ]
    logger.info(f"Training on {len(train_pairs)} valid preference pairs")

    # Train
    logger.info("Starting DPO training...")
    trainer.train(train_pairs)

    # Save final model
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
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
    logger.info("Starting DPO alignment")

    # Load and merge config
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    logger.info(f"Configuration: {config}")

    # Run training
    try:
        run_dpo(config)
        logger.info("DPO alignment completed successfully")
        return 0
    except Exception as e:
        logger.error(f"DPO alignment failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
