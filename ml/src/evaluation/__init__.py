"""
Persona Engine — Stage 6E End-to-End Evaluation & Benchmarking Package.
Exports evaluation schemas, benchmark cases, pipeline evaluator, metrics calculator, and comparison runner.
"""

from ml.src.evaluation.benchmark_cases import BENCHMARK_CASES, STANDARD_MEMORY_BANK
from ml.src.evaluation.comparison import ComparisonRunner
from ml.src.evaluation.evaluation_schema import (
    AblationMode,
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkResult,
    ErrorType,
    EvaluationSummary,
)
from ml.src.evaluation.evaluator import (
    EvaluationReport,
    EvaluationSample,
    PersonaEvaluator,
    PipelineEvaluator,
)
from ml.src.evaluation.metrics import (
    MetricsCalculator,
    calculate_emoji_consistency,
    calculate_length_similarity,
    calculate_punctuation_alignment,
    calculate_vocabulary_overlap,
    contains_emoji,
    contains_slang,
    detect_hallucination_indicators,
    is_hinglish_text,
)

__all__ = [
    "AblationMode",
    "BenchmarkCase",
    "BenchmarkCategory",
    "BenchmarkResult",
    "BENCHMARK_CASES",
    "ComparisonRunner",
    "ErrorType",
    "EvaluationReport",
    "EvaluationSample",
    "EvaluationSummary",
    "MetricsCalculator",
    "PersonaEvaluator",
    "PipelineEvaluator",
    "STANDARD_MEMORY_BANK",
    "calculate_emoji_consistency",
    "calculate_length_similarity",
    "calculate_punctuation_alignment",
    "calculate_vocabulary_overlap",
    "contains_emoji",
    "contains_slang",
    "detect_hallucination_indicators",
    "is_hinglish_text",
]
