"""
Evaluation module for the RLHF pipeline.

This module provides comprehensive evaluation tools:
- Benchmark integration (MT-Bench, AlpacaEval)
- Reward hacking detection
- Scaling analysis across model sizes
- Human preference evaluation framework

Modules:
    benchmarks: MT-Bench and AlpacaEval integration
    reward_hacking: Detection of reward gaming behaviors
    scaling: Analysis across model sizes
    human_eval: Human preference evaluation
"""

from evaluation.benchmarks import (
    BenchmarkRunner,
    MTBenchEvaluator,
    AlpacaEvalRunner,
    BenchmarkResult,
)
from evaluation.reward_hacking import (
    RewardHackingDetector,
    LengthExploitationDetector,
    SycophancyDetector,
    RepetitionDetector,
    HackingReport,
)
from evaluation.scaling import (
    ScalingAnalyzer,
    ScalingResult,
    plot_scaling_curves,
)
from evaluation.human_eval import (
    HumanEvalFramework,
    PreferenceCollector,
    AnnotationTask,
    EvaluationResult,
)

__all__ = [
    # Benchmarks
    "BenchmarkRunner",
    "MTBenchEvaluator",
    "AlpacaEvalRunner",
    "BenchmarkResult",
    # Reward Hacking
    "RewardHackingDetector",
    "LengthExploitationDetector",
    "SycophancyDetector",
    "RepetitionDetector",
    "HackingReport",
    # Scaling
    "ScalingAnalyzer",
    "ScalingResult",
    "plot_scaling_curves",
    # Human Eval
    "HumanEvalFramework",
    "PreferenceCollector",
    "AnnotationTask",
    "EvaluationResult",
]
