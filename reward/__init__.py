"""
Reward modeling module for the RLHF pipeline.

This module provides tools for training and analyzing reward models:
- Reward model architecture with scalar head on base LM
- Bradley-Terry preference training
- Reward distribution analysis and hacking detection

The reward model learns to predict human preferences from comparison
data and provides the training signal for PPO alignment.

Modules:
    reward_model: Reward model architecture and inference
    reward_trainer: Training loop for preference learning
    reward_analysis: Analysis tools for reward distributions
"""

from reward.reward_model import (
    RewardModel,
    RewardModelConfig,
    RewardHead,
)
from reward.reward_trainer import (
    RewardTrainer,
    RewardTrainingConfig,
    RewardTrainingCallback,
)
from reward.reward_analysis import (
    RewardAnalyzer,
    RewardDistribution,
    HackingDetector,
)

__all__ = [
    # Reward Model
    "RewardModel",
    "RewardModelConfig",
    "RewardHead",
    # Reward Trainer
    "RewardTrainer",
    "RewardTrainingConfig",
    "RewardTrainingCallback",
    # Reward Analysis
    "RewardAnalyzer",
    "RewardDistribution",
    "HackingDetector",
]
