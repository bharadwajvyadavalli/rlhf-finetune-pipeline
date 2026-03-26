"""
Data processing module for the RLHF fine-tuning pipeline.

This module provides utilities for:
- Data curation and quality filtering
- Synthetic data generation (Self-Instruct, Evol-Instruct)
- Preference pair generation for reward modeling and DPO
- Chat template formatting for various model architectures

Modules:
    curation: Data quality filtering, deduplication, and toxicity removal
    synthetic: Synthetic data generation using teacher models
    preference: Preference pair generation (chosen/rejected)
    formatting: Chat template formatting (ChatML, Alpaca, etc.)
"""

from data.curation import DataCurator, QualityFilter, Deduplicator, ToxicityFilter
from data.synthetic import SyntheticDataGenerator, SelfInstructGenerator, EvolInstructGenerator
from data.preference import PreferencePairGenerator, PreferenceDataset
from data.formatting import ChatTemplateFormatter, TemplateType

__all__ = [
    # Curation
    "DataCurator",
    "QualityFilter",
    "Deduplicator",
    "ToxicityFilter",
    # Synthetic
    "SyntheticDataGenerator",
    "SelfInstructGenerator",
    "EvolInstructGenerator",
    # Preference
    "PreferencePairGenerator",
    "PreferenceDataset",
    # Formatting
    "ChatTemplateFormatter",
    "TemplateType",
]
