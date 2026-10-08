"""
Comparative analysis engine for Stage 6E.
Executes baseline comparisons (Full Pipeline vs Context-Only Baseline) and
component ablation experiments (No Memory, No Context, No Orchestrator, Persona Only).
"""

from typing import Dict, List
from ml.src.evaluation.evaluation_schema import (
    AblationMode,
    BenchmarkCase,
    BenchmarkResult,
    EvaluationSummary,
)
from ml.src.evaluation.evaluator import PipelineEvaluator
from ml.src.evaluation.metrics import MetricsCalculator


class ComparisonRunner:
    """Runs baseline and ablation sweeps to evaluate subsystem contributions."""

    def __init__(self, evaluator: PipelineEvaluator):
        self.evaluator = evaluator

    def run_ablation_sweep(
        self,
        cases: List[BenchmarkCase],
    ) -> Dict[str, EvaluationSummary]:
        """
        Executes benchmark suite across all ablation configurations:
        - full_system
        - no_memory
        - no_context
        - no_orchestrator
        - persona_only
        """
        sweep_results: Dict[str, EvaluationSummary] = {}

        for mode in AblationMode:
            results = self.evaluator.evaluate_suite(cases, ablation_mode=mode)
            summary = MetricsCalculator.compute_summary(results)
            sweep_results[mode.value] = summary

        return sweep_results

    def compare_baseline_vs_full(
        self,
        cases: List[BenchmarkCase],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Directly compares Baseline (Persona Only) against Full Pipeline.
        Returns comparative metrics and delta values.
        """
        full_results = self.evaluator.evaluate_suite(cases, ablation_mode=AblationMode.FULL_SYSTEM)
        base_results = self.evaluator.evaluate_suite(cases, ablation_mode=AblationMode.PERSONA_ONLY)

        full_sum = MetricsCalculator.compute_summary(full_results)
        base_sum = MetricsCalculator.compute_summary(base_results)

        comparison_table = {
            "topic_accuracy": {
                "full_system": full_sum.topic_accuracy,
                "baseline": base_sum.topic_accuracy,
                "delta": round(full_sum.topic_accuracy - base_sum.topic_accuracy, 4),
            },
            "intent_accuracy": {
                "full_system": full_sum.intent_accuracy,
                "baseline": base_sum.intent_accuracy,
                "delta": round(full_sum.intent_accuracy - base_sum.intent_accuracy, 4),
            },
            "strategy_accuracy": {
                "full_system": full_sum.strategy_accuracy,
                "baseline": base_sum.strategy_accuracy,
                "delta": round(full_sum.strategy_accuracy - base_sum.strategy_accuracy, 4),
            },
            "ambiguity_accuracy": {
                "full_system": full_sum.ambiguity_accuracy,
                "baseline": base_sum.ambiguity_accuracy,
                "delta": round(full_sum.ambiguity_accuracy - base_sum.ambiguity_accuracy, 4),
            },
            "memory_precision": {
                "full_system": full_sum.memory_precision,
                "baseline": base_sum.memory_precision,
                "delta": round(full_sum.memory_precision - base_sum.memory_precision, 4),
            },
            "memory_recall": {
                "full_system": full_sum.memory_recall,
                "baseline": base_sum.memory_recall,
                "delta": round(full_sum.memory_recall - base_sum.memory_recall, 4),
            },
            "memory_contamination_rate": {
                "full_system": full_sum.memory_contamination_rate,
                "baseline": base_sum.memory_contamination_rate,
                "delta": round(full_sum.memory_contamination_rate - base_sum.memory_contamination_rate, 4),
            },
            "hinglish_rate": {
                "full_system": full_sum.hinglish_rate,
                "baseline": base_sum.hinglish_rate,
                "delta": round(full_sum.hinglish_rate - base_sum.hinglish_rate, 4),
            },
            "slang_rate": {
                "full_system": full_sum.slang_rate,
                "baseline": base_sum.slang_rate,
                "delta": round(full_sum.slang_rate - base_sum.slang_rate, 4),
            },
            "hallucination_rate": {
                "full_system": full_sum.hallucination_rate,
                "baseline": base_sum.hallucination_rate,
                "delta": round(full_sum.hallucination_rate - base_sum.hallucination_rate, 4),
            },
            "mean_total_latency_ms": {
                "full_system": full_sum.latency_summary["total_ms"]["mean"],
                "baseline": base_sum.latency_summary["total_ms"]["mean"],
                "delta": round(full_sum.latency_summary["total_ms"]["mean"] - base_sum.latency_summary["total_ms"]["mean"], 2),
            },
        }

        return comparison_table
