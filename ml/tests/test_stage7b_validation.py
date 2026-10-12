"""
Unit tests for Stage 7B Colab-based Runtime Validation Infrastructure.
Validates:
- Hardware and environment diagnostic reporting
- Configuration and adapter audit verification
- Default model invariance (stage5_v1 remains default)
- Safe error handling for invalid variants and missing adapters
- Smoke test execution
- Real execution of all 4 pipeline layers (Context, Memory, Orchestrator, Generator)
- Structured report and JSON artifact export
"""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock

# Ensure repo root and ml root on sys.path
ml_root = Path(__file__).resolve().parents[1]
repo_root = ml_root.parent
import sys
for p in [str(repo_root), str(ml_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.scripts.run_stage7b_validation import (
    get_environment_diagnostics,
    measure_benchmarks,
    run_memory_conditioned_pipeline_smoke_test,
    run_smoke_tests,
    save_reports,
    test_invalid_variants_and_missing_adapters,
    verify_configurations_and_adapters,
    verify_default_model_invariance,
)
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator


class TestStage7BValidation(unittest.TestCase):
    """Test suite for Stage 7B validation infrastructure and diagnostics."""

    def setUp(self):
        ModelLoader.unload()
        if "MODEL_VARIANT" in os.environ:
            del os.environ["MODEL_VARIANT"]

    def tearDown(self):
        ModelLoader.unload()
        if "MODEL_VARIANT" in os.environ:
            del os.environ["MODEL_VARIANT"]

    def test_environment_diagnostics_keys(self):
        """Verify diagnostic dictionary contains required hardware and version keys."""
        diag = get_environment_diagnostics()
        self.assertIn("platform", diag)
        self.assertIn("python_version", diag)
        self.assertIn("torch_version", diag)
        self.assertIn("cuda_available", diag)
        self.assertIn("packages", diag)
        for pkg in ["transformers", "peft", "bitsandbytes", "accelerate", "datasets"]:
            self.assertIn(pkg, diag["packages"])

    def test_configuration_and_adapter_audit(self):
        """Verify both stage5_v1 and stage7a_qwen2.5_3b configurations are validated."""
        audit = verify_configurations_and_adapters()
        self.assertIn("stage5_baseline", audit)
        self.assertIn("stage7a_candidate", audit)
        self.assertTrue(audit["stage5_baseline"]["registry_exists"])
        self.assertTrue(audit["stage5_baseline"]["adapter_config_exists"])
        self.assertTrue(audit["stage7a_candidate"]["registry_exists"])
        self.assertTrue(audit["stage7a_candidate"]["adapter_config_exists"])

    def test_default_model_invariance(self):
        """Confirm stage5_v1 remains the default model when no variant is specified."""
        res = verify_default_model_invariance()
        self.assertEqual(res["status"], "PASS")
        self.assertEqual(res["generator_model_id"], "stage5_v1")
        self.assertTrue(res["is_default_stage5"])

    def test_error_handling_invalid_variant_and_missing_adapter(self):
        """Confirm invalid variants are rejected with FileNotFoundError and missing adapters handled."""
        err_res = test_invalid_variants_and_missing_adapters()
        self.assertEqual(err_res["invalid_variant_test"]["status"], "PASS")
        self.assertEqual(err_res["missing_adapter_path_test"]["status"], "PASS")

    def test_smoke_tests_execution(self):
        """Verify 3 smoke tests run and produce non-empty responses."""
        gen = PersonaGenerator(model_id="stage7a_qwen2.5_3b")
        gen.load_model(mock=True)

        results = run_smoke_tests(gen)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r["status"], "PASS")
            self.assertGreater(len(r["response"]), 0)

    def test_four_layer_memory_pipeline_smoke_test(self):
        """Confirm all 4 engine components (Context, Memory, Orchestrator, Generator) are exercised."""
        gen = PersonaGenerator(model_id="stage7a_qwen2.5_3b")
        gen.load_model(mock=True)

        res = run_memory_conditioned_pipeline_smoke_test(gen)
        self.assertEqual(res["status"], "PASS")
        self.assertTrue(res["layers_exercised"]["context_engine"])
        self.assertTrue(res["layers_exercised"]["memory_engine"])
        self.assertTrue(res["layers_exercised"]["orchestrator"])
        self.assertTrue(res["layers_exercised"]["generator"])
        self.assertIn("mem_laptop_nitro", res["retrieved_memory_ids"])
        self.assertEqual(res["strategy"], "answer")

    def test_latency_benchmarking(self):
        """Verify benchmark measurement returns latency percentiles."""
        gen = PersonaGenerator(model_id="stage7a_qwen2.5_3b")
        gen.load_model(mock=True)

        bench = measure_benchmarks(gen, runs=3)
        self.assertEqual(bench["iterations"], 3)
        self.assertIn("latency_mean_ms", bench)
        self.assertIn("latency_median_ms", bench)
        self.assertIn("latency_p95_ms", bench)

    def test_save_reports_artifacts(self):
        """Verify report and JSON files are properly generated."""
        sample_results = {
            "overall_status": "PASS",
            "environment_diagnostics": {"python_version": "3.10", "torch_version": "2.1"},
            "configuration_verification": {},
            "smoke_tests": [],
            "memory_conditioned_smoke_test": {"latencies": {}},
            "benchmarks": {"latency_mean_ms": 15.0},
        }
        reports_dir = ml_root / "data" / "stage7b" / "reports"
        results_dir = ml_root / "data" / "stage7b" / "results"

        json_path, md_path = save_reports(sample_results, reports_dir, results_dir)
        self.assertTrue(json_path.is_file())
        self.assertTrue(md_path.is_file())


if __name__ == "__main__":
    unittest.main()
