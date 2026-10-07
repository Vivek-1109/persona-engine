"""
Persona Engine — Evaluation Subpackage
"""

from .evaluator import EvaluationReport, PersonaEvaluator
from .metrics import (
    calculate_emoji_consistency,
    calculate_length_similarity,
    calculate_punctuation_alignment,
    calculate_vocabulary_overlap,
)

__all__ = [
    "calculate_length_similarity",
    "calculate_vocabulary_overlap",
    "calculate_emoji_consistency",
    "calculate_punctuation_alignment",
    "PersonaEvaluator",
    "EvaluationReport",
]
