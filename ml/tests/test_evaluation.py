"""
Unit tests for Stage 6E End-to-End Evaluation & Benchmarking Module.
Validates benchmark cases integrity, metric calculations, evaluator pipeline,
ablation sweeps, and error categorization.
"""

import unittest

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
from ml.src.evaluation.evaluator import PipelineEvaluator
from ml.src.evaluation.metrics import (
    MetricsCalculator,
    contains_emoji,
    contains_slang,
    detect_hallucination_indicators,
    is_hinglish_text,
)


class TestEvaluationModule(unittest.TestCase):
    """Unit test suite for Stage 6E Evaluation module."""

    def setUp(self):
        self.evaluator = PipelineEvaluator()

    def test_benchmark_cases_count_and_coverage(self):
        """1. Verify at least 60 benchmark cases exist and all 16 categories are covered."""
        self.assertGreaterEqual(len(BENCHMARK_CASES), 60)
        covered_categories = {case.category for case in BENCHMARK_CASES}
        self.assertEqual(len(covered_categories), len(BenchmarkCategory))

    def test_multi_turn_depth_coverage(self):
        """2. Verify 1, 3, 5, 7, and 11 turn cases are present."""
        turn_counts = {len(case.conversation) for case in BENCHMARK_CASES}
        self.assertIn(1, turn_counts)
        self.assertIn(3, turn_counts)
        self.assertIn(5, turn_counts)
        self.assertIn(7, turn_counts)
        self.assertIn(11, turn_counts)

    def test_benchmark_case_schema_integrity(self):
        """3. Ensure all benchmark cases have valid structure and non-empty dialogues."""
        for case in BENCHMARK_CASES:
            self.assertTrue(case.case_id.startswith("CASE_"))
            self.assertGreater(len(case.conversation), 0)
            self.assertTrue(bool(case.expected_topic))
            self.assertTrue(bool(case.expected_intent))
            self.assertTrue(bool(case.expected_strategy))

    def test_evaluator_single_case_execution(self):
        """4. Verify end-to-end execution of a single benchmark case."""
        case = BENCHMARK_CASES[0]  # CASE_CASUAL_01
        result = self.evaluator.evaluate_case(case)
        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.case_id, case.case_id)
        self.assertTrue(result.topic_correct)
        self.assertGreater(result.latencies_ms["total_ms"], 0.0)
        self.assertIsInstance(result.final_response, str)

    def test_evaluator_gaming_case_memory_utilization(self):
        """5. Verify gaming invitation utilizes gaming memory."""
        gaming_case = next(c for c in BENCHMARK_CASES if c.case_id == "CASE_GAMING_01")
        result = self.evaluator.evaluate_case(gaming_case)
        self.assertEqual(result.predicted_topic, "gaming")
        self.assertEqual(result.predicted_strategy, "accept")
        self.assertTrue(result.use_memory)
        self.assertIn("mem_gaming_pref", result.selected_memory_ids)

    def test_evaluator_closing_case_memory_suppression(self):
        """6. Verify closing conversation suppresses memory usage."""
        close_case = next(c for c in BENCHMARK_CASES if c.case_id == "CASE_CLOSE_01")
        result = self.evaluator.evaluate_case(close_case)
        self.assertEqual(result.predicted_strategy, "close_conversation")
        self.assertFalse(result.use_memory)
        self.assertEqual(result.selected_memory_ids, [])

    def test_metrics_calculator_computation(self):
        """7. Verify MetricsCalculator correctly aggregates benchmark results."""
        subset = BENCHMARK_CASES[:10]
        results = self.evaluator.evaluate_suite(subset)
        summary = MetricsCalculator.compute_summary(results)
        self.assertIsInstance(summary, EvaluationSummary)
        self.assertEqual(summary.total_cases, 10)
        self.assertGreaterEqual(summary.topic_accuracy, 0.0)
        self.assertLessEqual(summary.topic_accuracy, 1.0)
        self.assertIn("mean", summary.length_distribution)
        self.assertIn("total_ms", summary.latency_summary)

    def test_stylistic_proxy_helpers(self):
        """8. Test Hinglish, slang, emoji, and hallucination heuristic detectors."""
        self.assertTrue(is_hinglish_text("kya chal raha hai bhai"))
        self.assertFalse(is_hinglish_text("the quick brown fox"))

        self.assertTrue(contains_slang("scene kya hai bro"))
        self.assertFalse(contains_slang("good morning sir"))

        self.assertTrue(contains_emoji("great game! \U0001f3ae"))
        self.assertFalse(contains_emoji("great game!"))

        self.assertTrue(detect_hallucination_indicators("Your ASUS battery is damaged", "technology"))
        self.assertFalse(detect_hallucination_indicators("Check your battery settings", "technology"))

    def test_ablation_sweep_execution(self):
        """9. Verify ComparisonRunner runs ablation configurations successfully."""
        subset = BENCHMARK_CASES[:3]
        runner = ComparisonRunner(self.evaluator)
        sweep = runner.run_ablation_sweep(subset)
        self.assertIn("full_system", sweep)
        self.assertIn("no_memory", sweep)
        self.assertIn("persona_only", sweep)
        self.assertEqual(sweep["full_system"].total_cases, 3)

    def test_baseline_vs_full_comparison(self):
        """10. Verify direct baseline comparison table generation."""
        subset = BENCHMARK_CASES[:3]
        runner = ComparisonRunner(self.evaluator)
        table = runner.compare_baseline_vs_full(subset)
        self.assertIn("strategy_accuracy", table)
        self.assertIn("delta", table["strategy_accuracy"])


if __name__ == "__main__":
    unittest.main()
