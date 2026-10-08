"""
Benchmark runner script for Stage 6E End-to-End Evaluation.
Executes the complete 65-case benchmark suite across Full Pipeline, Baseline, and Ablation modes.
Generates structured JSON and JSONL artifacts in ml/data/stage5/evaluation/.
"""

import json
from pathlib import Path
import sys
import time

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.evaluation.benchmark_cases import BENCHMARK_CASES
from ml.src.evaluation.comparison import ComparisonRunner
from ml.src.evaluation.evaluation_schema import (
    AblationMode,
    BenchmarkResult,
    EvaluationSummary,
)
from ml.src.evaluation.evaluator import PipelineEvaluator
from ml.src.evaluation.metrics import MetricsCalculator


def main():
    print("=" * 70)
    print("STAGE 6E -- END-TO-END PIPELINE EVALUATION & BENCHMARKING")
    print("=" * 70)
    print(f"Total benchmark test cases: {len(BENCHMARK_CASES)}")

    output_dir = REPO_ROOT / "ml" / "data" / "stage5" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = PipelineEvaluator()
    comparison_runner = ComparisonRunner(evaluator)

    # 1. Warm-up run
    print("\n[Phase 1] Executing warm-up run...")
    evaluator.evaluate_case(BENCHMARK_CASES[0])

    # 2. Full System Evaluation
    print("\n[Phase 2] Running Full Pipeline on all 65 benchmark cases...")
    t_start = time.time()
    results = evaluator.evaluate_suite(BENCHMARK_CASES, ablation_mode=AblationMode.FULL_SYSTEM)
    full_duration = time.time() - t_start
    summary = MetricsCalculator.compute_summary(results)

    print(f"Full pipeline evaluated in {full_duration:.2f}s.")
    print("-" * 50)
    print(f"Topic Accuracy:       {summary.topic_accuracy * 100:.1f}%")
    print(f"Intent Accuracy:      {summary.intent_accuracy * 100:.1f}%")
    print(f"Strategy Accuracy:    {summary.strategy_accuracy * 100:.1f}%")
    print(f"Ambiguity Accuracy:   {summary.ambiguity_accuracy * 100:.1f}%")
    print(f"Memory Precision:     {summary.memory_precision * 100:.1f}%")
    print(f"Memory Recall:        {summary.memory_recall * 100:.1f}%")
    print(f"Memory Contamination: {summary.memory_contamination_rate * 100:.1f}%")
    print(f"Hinglish Rate:        {summary.hinglish_rate * 100:.1f}%")
    print(f"Slang Rate:           {summary.slang_rate * 100:.1f}%")
    print(f"Hallucination Rate:   {summary.hallucination_rate * 100:.1f}%")
    print(f"Mean Total Latency:   {summary.latency_summary['total_ms']['mean']:.2f} ms")
    print(f"P95 Total Latency:    {summary.latency_summary['total_ms']['p95']:.2f} ms")

    # 3. Baseline Comparison (Full vs Baseline)
    print("\n[Phase 3] Running Baseline Comparison (Persona-Only vs Full Pipeline)...")
    comparison_table = comparison_runner.compare_baseline_vs_full(BENCHMARK_CASES)
    print("-" * 50)
    for metric_name, vals in comparison_table.items():
        delta_sign = "+" if vals["delta"] >= 0 else ""
        print(f"  {metric_name:26}: Full={vals['full_system']:<7} | Base={vals['baseline']:<7} | Delta={delta_sign}{vals['delta']}")

    # 4. Ablation Sweep
    print("\n[Phase 4] Running Component Ablation Sweep...")
    ablation_sweep = comparison_runner.run_ablation_sweep(BENCHMARK_CASES)
    print("-" * 50)
    for mode_name, ab_sum in ablation_sweep.items():
        print(f"  {mode_name:18}: StratAcc={ab_sum.strategy_accuracy*100:.1f}% | MemRecall={ab_sum.memory_recall*100:.1f}% | Latency={ab_sum.latency_summary['total_ms']['mean']:.1f}ms")

    # 5. Determinism Verification
    print("\n[Phase 5] Verifying Determinism on Reproducibility Subset...")
    subset = BENCHMARK_CASES[:5]
    run_1 = [r.to_dict() for r in evaluator.evaluate_suite(subset)]
    run_2 = [r.to_dict() for r in evaluator.evaluate_suite(subset)]
    run_3 = [r.to_dict() for r in evaluator.evaluate_suite(subset)]

    # Check structural consistency (topic, intent, strategy, memory)
    def extract_structure(run_data):
        return [(r["case_id"], r["predicted_topic"], r["predicted_strategy"], r["selected_memory_ids"]) for r in run_data]

    deterministic = (extract_structure(run_1) == extract_structure(run_2) == extract_structure(run_3))
    print(f"Deterministic Consistency (3 runs): {'PASS (100% Identical)' if deterministic else 'FAIL'}")

    # 6. Save JSON and JSONL artifacts
    print("\n[Phase 6] Saving Structured Results...")
    results_json_path = output_dir / "stage6e_results.json"
    results_jsonl_path = output_dir / "stage6e_results.jsonl"
    summary_json_path = output_dir / "stage6e_summary.json"

    # JSON results
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # JSONL results
    with open(results_jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")

    # Summary with comparison & ablations
    full_summary_payload = {
        "summary": summary.to_dict(),
        "baseline_comparison": comparison_table,
        "ablation_sweep": {k: v.to_dict() for k, v in ablation_sweep.items()},
        "deterministic": deterministic,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary_payload, f, indent=2)

    print(f"  Artifact saved: {results_json_path}")
    print(f"  Artifact saved: {results_jsonl_path}")
    print(f"  Artifact saved: {summary_json_path}")

    print("\n" + "=" * 70)
    print("STAGE 6E EVALUATION RUN COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
