"""
Alignment module for the RLHF pipeline.

This module provides implementations of alignment algorithms:
- PPO (Proximal Policy Optimization) for RLHF
- DPO (Direct Preference Optimization) as an alternative
- Head-to-head comparison tools

Alignment is the final stage that adapts the instruction-following
model to better match human preferences.

Modules:
    ppo_trainer: PPO implementation with KL penalty and GAE
    dpo_trainer: Direct Preference Optimization
    comparison: PPO vs DPO evaluation framework
"""

from alignment.ppo_trainer import (
    PPOTrainer,
    PPOConfig,
    PPOStats,
)
from alignment.dpo_trainer import (
    DPOTrainer,
    DPOConfig,
    DPOLossType,
)
from alignment.comparison import (
    AlignmentComparator,
    ComparisonResult,
    compare_models,
)

__all__ = [
    # PPO
    "PPOTrainer",
    "PPOConfig",
    "PPOStats",
    # DPO
    "DPOTrainer",
    "DPOConfig",
    "DPOLossType",
    # Comparison
    "AlignmentComparator",
    "ComparisonResult",
    "compare_models",
]
