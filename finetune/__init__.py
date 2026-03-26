"""
Supervised Fine-Tuning (SFT) module for the RLHF pipeline.

This module provides tools for fine-tuning base language models on
instruction-following data using parameter-efficient methods:

- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- QLoRA for 4-bit quantized training
- Custom data collators for causal language modeling
- Multi-GPU training support with DeepSpeed

Modules:
    sft_trainer: Main SFT training logic with HuggingFace Trainer
    lora_config: LoRA/QLoRA configuration and setup
    data_collator: Custom collators for chat and instruction data
"""

from finetune.sft_trainer import SFTTrainer, SFTConfig
from finetune.lora_config import LoRAConfig, QLoRAConfig, create_peft_config
from finetune.data_collator import (
    DataCollatorForCausalLM,
    DataCollatorForCompletionOnly,
    DataCollatorForChat,
)

__all__ = [
    # SFT Trainer
    "SFTTrainer",
    "SFTConfig",
    # LoRA Config
    "LoRAConfig",
    "QLoRAConfig",
    "create_peft_config",
    # Data Collators
    "DataCollatorForCausalLM",
    "DataCollatorForCompletionOnly",
    "DataCollatorForChat",
]
