#!/usr/bin/env python3
"""
Run Supervised Fine-Tuning (SFT) training.

This script runs the SFT stage of the pipeline, fine-tuning a base
language model on instruction-following data using LoRA/QLoRA.

Usage:
    python scripts/run_sft.py --config configs/sft_config.yaml
    python scripts/run_sft.py --model meta-llama/Llama-2-7b-hf --dataset tatsu-lab/alpaca
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.formatting import ChatTemplateFormatter
from finetune.lora_config import get_lora_config, get_qlora_config
from finetune.sft_trainer import SFTTrainer, SFTConfig


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
        description="Run Supervised Fine-Tuning (SFT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_config.yaml",
        help="Path to configuration YAML file",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (overrides config)",
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
        help="Dataset name or path (overrides config)",
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
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Warmup steps (overrides config)",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank (overrides config)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (overrides config)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout (overrides config)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization)",
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

    Command-line arguments take precedence.

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
    if "lora" not in config:
        config["lora"] = {}

    # Model arguments
    if args.model:
        config["model"]["name"] = args.model
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

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
    if args.warmup_steps:
        config["training"]["warmup_steps"] = args.warmup_steps

    # LoRA arguments
    if args.no_lora:
        config["lora"]["enabled"] = False
    else:
        config["lora"]["enabled"] = config.get("lora", {}).get("enabled", True)
    if args.lora_r:
        config["lora"]["r"] = args.lora_r
    if args.lora_alpha:
        config["lora"]["alpha"] = args.lora_alpha
    if args.lora_dropout:
        config["lora"]["dropout"] = args.lora_dropout
    if args.qlora:
        config["lora"]["use_qlora"] = True

    # Other arguments
    if args.seed:
        config["training"]["seed"] = args.seed
    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume
    if args.wandb_project:
        config["training"]["wandb_project"] = args.wandb_project

    # Set defaults
    config["model"].setdefault("name", "EleutherAI/pythia-410m")
    config["data"].setdefault("dataset", "tatsu-lab/alpaca")
    config["data"].setdefault("max_seq_length", 2048)
    config["training"].setdefault("output_dir", "outputs/sft")
    config["training"].setdefault("num_epochs", 3)
    config["training"].setdefault("per_device_batch_size", 4)
    config["training"].setdefault("learning_rate", 2e-5)
    config["training"].setdefault("gradient_accumulation_steps", 4)
    config["training"].setdefault("warmup_steps", 100)
    config["training"].setdefault("seed", 42)
    config["lora"].setdefault("r", 16)
    config["lora"].setdefault("alpha", 32)
    config["lora"].setdefault("dropout", 0.05)
    config["lora"].setdefault("use_qlora", False)

    return config


def run_sft(config: dict[str, Any]) -> None:
    """Run SFT training with the given configuration.

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

    # Load model
    logger.info(f"Loading model from {model_name}")
    device_map = "auto" if torch.cuda.is_available() else None

    # Get quantization config if using QLoRA
    quantization_config = None
    if config["lora"].get("use_qlora", False):
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using QLoRA with 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    # Get LoRA config
    lora_config = None
    if config["lora"].get("enabled", True):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"].get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info(f"Applied LoRA with r={config['lora']['r']}, alpha={config['lora']['alpha']}")

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

    # Format dataset
    formatter = ChatTemplateFormatter(model_name=model_name)

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        # Handle different dataset formats
        if "instruction" in example and "output" in example:
            text = formatter.format(
                instruction=example["instruction"],
                response=example["output"],
                input_text=example.get("input", ""),
            )
        elif "prompt" in example and "response" in example:
            text = formatter.format(
                instruction=example["prompt"],
                response=example["response"],
            )
        elif "text" in example:
            text = example["text"]
        else:
            # Try to find suitable columns
            keys = list(example.keys())
            text = str(example.get(keys[0], ""))

        return {"text": text}

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    logger.info(f"Formatted {len(train_dataset)} training examples")

    # Create SFT config
    sft_config = SFTConfig(
        model_name=model_name,
        output_dir=config["training"]["output_dir"],
        num_epochs=config["training"]["num_epochs"],
        per_device_batch_size=config["training"]["per_device_batch_size"],
        learning_rate=config["training"]["learning_rate"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        max_seq_length=config["data"]["max_seq_length"],
        use_lora=config["lora"].get("enabled", True),
        use_qlora=config["lora"].get("use_qlora", False),
        seed=seed,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=sft_config,
        train_dataset=train_dataset,
    )

    # Train
    logger.info("Starting training...")
    resume_from = config["training"].get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    output_dir = Path(config["training"]["output_dir"])
    trainer.save(output_dir / "final")
    logger.info(f"Saved final model to {output_dir / 'final'}")

    # Optionally merge LoRA weights
    if config["lora"].get("enabled", True) and config["lora"].get("merge_on_save", False):
        logger.info("Merging LoRA weights into base model...")
        trainer.merge_and_save(output_dir / "merged")
        logger.info(f"Saved merged model to {output_dir / 'merged'}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code
    """
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting SFT training")

    # Load and merge config
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    logger.info(f"Configuration: {config}")

    # Run training
    try:
        run_sft(config)
        logger.info("SFT training completed successfully")
        return 0
    except Exception as e:
        logger.error(f"SFT training failed: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())
