"""
LoRA and QLoRA configuration management.

This module provides configuration classes and utilities for setting up
parameter-efficient fine-tuning with LoRA (Low-Rank Adaptation) and
QLoRA (Quantized LoRA).

LoRA adds trainable low-rank decomposition matrices to attention layers,
dramatically reducing the number of trainable parameters while maintaining
performance close to full fine-tuning.

QLoRA extends this by using 4-bit quantization of the base model, enabling
fine-tuning of large models on consumer GPUs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from transformers import PreTrainedModel

# Lazy imports
_PEFT_AVAILABLE = False
_BNB_AVAILABLE = False

try:
    from peft import LoraConfig as PEFTLoraConfig
    from peft import TaskType as PEFTTaskType
    from peft import get_peft_model, prepare_model_for_kbit_training as peft_prepare
    from peft import PeftModel
    _PEFT_AVAILABLE = True
except ImportError:
    pass

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    pass


class QuantizationType(Enum):
    """Types of quantization available."""

    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    FP4 = "fp4"


class TaskType(Enum):
    """PEFT task types."""

    CAUSAL_LM = "CAUSAL_LM"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    SEQ_CLS = "SEQ_CLS"
    TOKEN_CLS = "TOKEN_CLS"


# Model-specific target modules
MODEL_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama-attention-only": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "pythia": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "pythia-attention-only": ["query_key_value"],
    "gpt-neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "mpt": ["Wqkv", "out_proj", "up_proj", "down_proj"],
    "bloom": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
}


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation).

    Attributes:
        r: Rank of the low-rank decomposition. Higher values mean more
           parameters and capacity, but also more memory. Typical: 8-64
        lora_alpha: Scaling factor for LoRA. The effective scaling is
                   lora_alpha / r. Common to set alpha = r or alpha = 2*r
        lora_dropout: Dropout probability for LoRA layers. Helps prevent
                     overfitting, especially on small datasets
        target_modules: List of module names to apply LoRA to. For attention,
                       typically ["q_proj", "v_proj"] or all projection layers
        bias: Bias training strategy: "none", "all", or "lora_only"
        task_type: PEFT task type for the model
        modules_to_save: Additional modules to train and save (not LoRA)
        fan_in_fan_out: Set True for Conv1D layers (GPT-2 style)
        init_lora_weights: How to initialize LoRA weights
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM
    modules_to_save: list[str] | None = None
    fan_in_fan_out: bool = False
    init_lora_weights: bool | str = True

    def to_peft_config(self) -> Any:
        """Convert to PEFT LoraConfig object.

        Returns:
            peft.LoraConfig instance
        """
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required for LoRA. Install with: pip install peft")

        # Map TaskType to PEFT TaskType
        task_type_map = {
            TaskType.CAUSAL_LM: PEFTTaskType.CAUSAL_LM,
            TaskType.SEQ_2_SEQ_LM: PEFTTaskType.SEQ_2_SEQ_LM,
            TaskType.SEQ_CLS: PEFTTaskType.SEQ_CLS,
            TaskType.TOKEN_CLS: PEFTTaskType.TOKEN_CLS,
        }

        return PEFTLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=task_type_map.get(self.task_type, PEFTTaskType.CAUSAL_LM),
            modules_to_save=self.modules_to_save,
            fan_in_fan_out=self.fan_in_fan_out,
            init_lora_weights=self.init_lora_weights,
        )

    def get_trainable_params_info(self, model: PreTrainedModel) -> dict[str, Any]:
        """Get information about trainable parameters.

        Args:
            model: Model with LoRA applied

        Returns:
            Dictionary with parameter counts and percentages
        """
        trainable_params = 0
        all_params = 0

        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percent": 100 * trainable_params / all_params if all_params > 0 else 0,
        }


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (Quantized LoRA).

    Extends LoRAConfig with quantization settings for 4-bit training.
    Enables fine-tuning of large models on consumer GPUs.

    Additional Attributes:
        load_in_4bit: Enable 4-bit quantization
        load_in_8bit: Enable 8-bit quantization (alternative)
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_compute_dtype: Compute dtype for 4-bit operations
        bnb_4bit_use_double_quant: Use nested quantization for more savings
        llm_int8_threshold: Threshold for mixed int8/fp16 (8-bit mode)
        llm_int8_has_fp16_weight: Keep fp16 weights for some layers
    """

    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False

    def to_bnb_config(self) -> Any:
        """Convert to BitsAndBytesConfig object.

        Returns:
            BitsAndBytesConfig instance
        """
        if not _BNB_AVAILABLE:
            raise ImportError(
                "bitsandbytes is required for QLoRA. Install with: pip install bitsandbytes"
            )

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            load_in_8bit=self.load_in_8bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.get_compute_dtype(),
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            llm_int8_threshold=self.llm_int8_threshold,
            llm_int8_has_fp16_weight=self.llm_int8_has_fp16_weight,
        )

    def get_compute_dtype(self) -> torch.dtype:
        """Get the compute dtype as torch dtype.

        Returns:
            torch.dtype for computation
        """
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return dtype_map.get(self.bnb_4bit_compute_dtype.lower(), torch.bfloat16)


def create_peft_config(
    lora_config: LoRAConfig | QLoRAConfig,
) -> Any:
    """Create a PEFT configuration from LoRAConfig.

    Args:
        lora_config: LoRA or QLoRA configuration

    Returns:
        PEFT LoraConfig object ready for use
    """
    return lora_config.to_peft_config()


def create_quantization_config(
    qlora_config: QLoRAConfig,
) -> Any:
    """Create a BitsAndBytesConfig from QLoRAConfig.

    Args:
        qlora_config: QLoRA configuration

    Returns:
        BitsAndBytesConfig object
    """
    return qlora_config.to_bnb_config()


def get_target_modules_for_model(
    model_name: str,
    include_all_linear: bool = False,
) -> list[str]:
    """Get recommended target modules for a model architecture.

    Args:
        model_name: Name or path of the model
        include_all_linear: Whether to include all linear layers

    Returns:
        List of module names to target
    """
    model_lower = model_name.lower()

    # Try to match model architecture
    if "llama" in model_lower or "llama-2" in model_lower or "llama2" in model_lower:
        key = "llama" if include_all_linear else "llama-attention-only"
        return MODEL_TARGET_MODULES.get(key, MODEL_TARGET_MODULES["llama"])

    if "mistral" in model_lower or "mixtral" in model_lower:
        return MODEL_TARGET_MODULES.get("mistral", MODEL_TARGET_MODULES["llama"])

    if "pythia" in model_lower or "gpt-neox" in model_lower:
        key = "pythia" if include_all_linear else "pythia-attention-only"
        return MODEL_TARGET_MODULES.get(key, MODEL_TARGET_MODULES["pythia"])

    if "gpt2" in model_lower:
        return MODEL_TARGET_MODULES.get("gpt2", ["c_attn", "c_proj"])

    if "falcon" in model_lower:
        return MODEL_TARGET_MODULES.get("falcon", MODEL_TARGET_MODULES["pythia"])

    if "mpt" in model_lower:
        return MODEL_TARGET_MODULES.get("mpt", ["Wqkv", "out_proj"])

    if "bloom" in model_lower:
        return MODEL_TARGET_MODULES.get("bloom", MODEL_TARGET_MODULES["pythia"])

    if "opt" in model_lower:
        return MODEL_TARGET_MODULES.get("opt", ["q_proj", "v_proj"])

    if "phi" in model_lower:
        return MODEL_TARGET_MODULES.get("phi", ["q_proj", "k_proj", "v_proj", "dense"])

    # Default to Llama-style modules
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def prepare_model_for_kbit_training(
    model: PreTrainedModel,
    use_gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """Prepare a quantized model for training.

    Handles necessary modifications for training quantized models:
    - Enables gradient computation on quantized weights
    - Sets up gradient checkpointing
    - Configures layer norms for training

    Args:
        model: Quantized model
        use_gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        Model ready for k-bit training
    """
    if not _PEFT_AVAILABLE:
        raise ImportError("peft is required. Install with: pip install peft")

    model = peft_prepare(model, use_gradient_checkpointing=use_gradient_checkpointing)

    # Ensure embedding layers are in float32 for stability
    for name, module in model.named_modules():
        if "norm" in name.lower():
            module = module.to(torch.float32)

    return model


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """Print the number of trainable vs total parameters.

    Useful for verifying LoRA is set up correctly.

    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_params:,} || "
        f"trainable%: {100 * trainable_params / all_params:.2f}"
    )


def merge_lora_weights(
    model: PreTrainedModel,
    output_path: str | None = None,
) -> PreTrainedModel:
    """Merge LoRA weights into the base model.

    Creates a standalone model with LoRA weights merged in,
    useful for deployment without PEFT dependency.

    Args:
        model: Model with LoRA adapters
        output_path: Optional path to save merged model

    Returns:
        Model with merged weights
    """
    if not _PEFT_AVAILABLE:
        raise ImportError("peft is required. Install with: pip install peft")

    if hasattr(model, 'merge_and_unload'):
        merged_model = model.merge_and_unload()
    else:
        raise ValueError("Model does not have LoRA adapters to merge")

    if output_path:
        merged_model.save_pretrained(output_path)

    return merged_model


def load_lora_weights(
    model: PreTrainedModel,
    lora_path: str,
    adapter_name: str = "default",
) -> PreTrainedModel:
    """Load LoRA weights into a model.

    Args:
        model: Base model
        lora_path: Path to saved LoRA weights
        adapter_name: Name for the adapter

    Returns:
        Model with loaded LoRA weights
    """
    if not _PEFT_AVAILABLE:
        raise ImportError("peft is required. Install with: pip install peft")

    model = PeftModel.from_pretrained(
        model,
        lora_path,
        adapter_name=adapter_name,
    )

    return model


@dataclass
class LoRALayerInfo:
    """Information about a LoRA layer.

    Attributes:
        name: Name of the layer
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        alpha: LoRA alpha
        params: Number of LoRA parameters
    """

    name: str
    in_features: int
    out_features: int
    rank: int
    alpha: int
    params: int


def analyze_lora_layers(model: PreTrainedModel) -> list[LoRALayerInfo]:
    """Analyze LoRA layers in a model.

    Args:
        model: Model with LoRA adapters

    Returns:
        List of LoRALayerInfo for each LoRA layer
    """
    layers = []

    for name, module in model.named_modules():
        # Check if this is a LoRA layer
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Get dimensions from LoRA matrices
            lora_a = module.lora_A.default
            lora_b = module.lora_B.default

            in_features = lora_a.weight.shape[1]
            rank = lora_a.weight.shape[0]
            out_features = lora_b.weight.shape[0]

            # Get alpha if available
            alpha = getattr(module, 'scaling', {}).get('default', rank)
            if hasattr(module, 'lora_alpha'):
                alpha = module.lora_alpha.get('default', rank)

            # Calculate parameters
            params = lora_a.weight.numel() + lora_b.weight.numel()

            layers.append(LoRALayerInfo(
                name=name,
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha if isinstance(alpha, int) else rank,
                params=params,
            ))

    return layers


def get_lora_config(
    model_name: str,
    use_qlora: bool = False,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    include_all_linear: bool = False,
) -> LoRAConfig | QLoRAConfig:
    """Create a LoRA or QLoRA config for a specific model.

    Convenience function that auto-detects target modules.

    Args:
        model_name: Name of the model
        use_qlora: Whether to use QLoRA
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        include_all_linear: Include all linear layers

    Returns:
        Configured LoRAConfig or QLoRAConfig
    """
    target_modules = get_target_modules_for_model(model_name, include_all_linear)

    if use_qlora:
        return QLoRAConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )
    else:
        return LoRAConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )
