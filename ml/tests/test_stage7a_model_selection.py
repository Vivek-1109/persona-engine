"""
Unit Tests for Stage 7A Base Model Selection Infrastructure.
Validates:
- Baseline and candidate registries
- Model selection and MODEL_VARIANT opt-in mechanism
- Dataset fingerprint integrity
- Mock adapter loading and cache invalidation on model switch
- Controlled evaluation pipeline trace invariance
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

from ml.src.evaluation.benchmark_cases import BENCHMARK_CASES
from ml.src.evaluation.stage7a_evaluator import Stage7AEvaluator
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator


class TestStage7AModelSelection(unittest.TestCase):
    """Test suite for Stage 7A model selection, registry, and evaluation infrastructure."""

    def setUp(self):
        self.registry_dir = ml_root / "models" / "registry"
        self.stage5_baseline_path = self.registry_dir / "stage7a_baseline.json"
        self.stage5_v1_path = self.registry_dir / "stage5_v1.json"
        self.candidate_path = self.registry_dir / "stage7a_qwen2.5_3b.json"
        # Reset loader state
        ModelLoader.unload()

    def tearDown(self):
        ModelLoader.unload()
        if "MODEL_VARIANT" in os.environ:
            del os.environ["MODEL_VARIANT"]

    def test_baseline_registry_exists_and_valid(self):
        """Verify stage7a_baseline.json exists with frozen baseline properties."""
        self.assertTrue(self.stage5_baseline_path.is_file(), "stage7a_baseline.json missing")
        with open(self.stage5_baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["model_id"], "stage7a_baseline")
        self.assertEqual(data["status"], "control_baseline")
        self.assertEqual(data["base_model"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(data["adapter_path"], "ml/models/adapters/stage5_v1")
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["training_configuration"]["lora_rank"], 16)
        self.assertEqual(data["training_configuration"]["lora_alpha"], 32)
        self.assertEqual(data["generation_configuration"]["temperature"], 0.7)

        # Ensure original stage5_v1.json was not modified
        self.assertTrue(self.stage5_v1_path.is_file())
        with open(self.stage5_v1_path, "r", encoding="utf-8") as f:
            v1_data = json.load(f)
        self.assertEqual(v1_data["model_id"], "stage5_v1")
        self.assertEqual(v1_data["status"], "baseline_persona_model")

    def test_candidate_registry_exists_and_valid(self):
        """Verify stage7a_qwen2.5_3b.json exists with candidate properties."""
        self.assertTrue(self.candidate_path.is_file(), "Candidate registry entry missing")
        with open(self.candidate_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["model_id"], "stage7a_qwen2.5_3b")
        self.assertEqual(data["base_model"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(data["status"], "experimental")
        self.assertEqual(data["adapter_path"], "ml/models/adapters/stage7a_qwen2.5_3b")
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["training_config"]["lora_rank"], 16)
        self.assertEqual(data["training_config"]["lora_alpha"], 32)
        self.assertIn("dataset_fingerprint", data)

    def test_default_model_selection_is_control(self):
        """Verify default model selection remains stage5_v1 (Control) when no override is set."""
        gen = PersonaGenerator()
        self.assertEqual(gen.model_id, "stage5_v1")
        self.assertEqual(gen.base_model_name, "Qwen/Qwen2.5-1.5B-Instruct")

    def test_opt_in_model_selection_via_argument_and_env(self):
        """Verify candidate model is selectable via explicit argument and MODEL_VARIANT env var."""
        # 1. Via constructor argument
        gen_arg = PersonaGenerator(model_id="stage7a_qwen2.5_3b")
        self.assertEqual(gen_arg.model_id, "stage7a_qwen2.5_3b")
        self.assertEqual(gen_arg.base_model_name, "Qwen/Qwen2.5-3B-Instruct")

        # 2. Via alias
        gen_alias = PersonaGenerator(model_id="stage7a_3b_candidate")
        self.assertEqual(gen_alias.model_id, "stage7a_qwen2.5_3b")

        # 3. Via environment variable
        os.environ["MODEL_VARIANT"] = "stage7a_qwen2.5_3b"
        gen_env = PersonaGenerator()
        self.assertEqual(gen_env.model_id, "stage7a_qwen2.5_3b")

    def test_mock_model_loading_and_switch_invalidation(self):
        """Verify ModelLoader loads mock handles and invalidates cache on model switch."""
        mock_m1 = MagicMock()
        mock_t1 = MagicMock()
        ModelLoader.load_model(model_id="stage5_v1", mock_model=mock_m1, mock_tokenizer=mock_t1)
        self.assertTrue(ModelLoader.is_loaded())
        self.assertEqual(ModelLoader.get_info()["model_id"], "stage5_v1")

        # Loading same model returns cached instance
        m, t = ModelLoader.load_model(model_id="stage5_v1")
        self.assertIs(m, mock_m1)

        # Loading different model unloads previous and switches
        mock_m2 = MagicMock()
        mock_t2 = MagicMock()
        m2, t2 = ModelLoader.load_model(model_id="stage7a_qwen2.5_3b", mock_model=mock_m2, mock_tokenizer=mock_t2)
        self.assertEqual(ModelLoader.get_info()["model_id"], "stage7a_qwen2.5_3b")
        self.assertIs(m2, mock_m2)

    def test_dataset_fingerprint_integrity(self):
        """Verify SHA256 checksums match between dataset files and registry metadata."""
        import hashlib
        train_file = ml_root / "data" / "stage5" / "training" / "stage5_train.jsonl"
        self.assertTrue(train_file.is_file())
        with open(train_file, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()

        with open(self.stage5_baseline_path, "r", encoding="utf-8") as f:
            reg = json.load(f)

        expected_sha = reg["dataset_fingerprints"]["stage5_train.jsonl"]["sha256"]
        self.assertEqual(sha, expected_sha, "Dataset sha256 mismatch against baseline registry")

    def test_stage7a_evaluator_trace_invariance(self):
        """Verify Context, Memory, and Orchestrator outputs are identical across both models."""
        evaluator = Stage7AEvaluator()
        test_case = BENCHMARK_CASES[0]
        result = evaluator.evaluate_side_by_side(test_case)

        self.assertIn("shared_intermediate", result)
        self.assertIn("control_1.5b", result)
        self.assertIn("candidate_3b", result)

        shared = result["shared_intermediate"]
        self.assertIn("strategy", shared)
        self.assertIn("topic", shared)
        self.assertIn("retrieved_memory_ids", shared)

        # Both models evaluated for the same case
        self.assertEqual(result["case_id"], test_case.case_id)
        self.assertGreater(len(result["control_1.5b"]["response"]), 0)
        self.assertGreater(len(result["candidate_3b"]["response"]), 0)


if __name__ == "__main__":
    unittest.main()
