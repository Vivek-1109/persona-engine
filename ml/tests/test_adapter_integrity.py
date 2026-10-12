"""
Unit tests for Stage 7B.2: Strict Adapter Loading & Validation Integrity.

Verifies:
1. Missing adapter directory raises FileNotFoundError (no silent fallback).
2. Missing adapter config raises FileNotFoundError.
3. Corrupt adapter config raises ValueError.
4. Missing adapter weights raises FileNotFoundError.
5. Empty (0-byte) adapter weights raises FileNotFoundError.
6. Failed PEFT adapter loading raises RuntimeError (no silent base model fallback).
7. Successful adapter loading attaches PeftModel and reports adapter_loaded=True.
8. Candidate 3B adapter loading correctly verifies 3B base model and adapter metadata.
9. Invalid variant IDs are rejected with FileNotFoundError.
10. Baseline default invariance preserves stage5_v1.
11. Intentional mock behavior is isolated (is_mock=True, adapter_loaded=False).
"""

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure repo root and ml root on sys.path
ml_root = Path(__file__).resolve().parents[1]
repo_root = ml_root.parent
import sys
for p in [str(repo_root), str(ml_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator


class TestAdapterLoadingIntegrity(unittest.TestCase):
    """Strict tests for adapter loading, error boundaries, and metadata integrity."""

    def setUp(self):
        ModelLoader.unload()
        self.temp_dir = tempfile.mkdtemp()
        self.old_override = os.environ.get("ADAPTER_OVERRIDE_PATH")
        self.old_variant = os.environ.get("MODEL_VARIANT")
        if "ADAPTER_OVERRIDE_PATH" in os.environ:
            del os.environ["ADAPTER_OVERRIDE_PATH"]
        if "MODEL_VARIANT" in os.environ:
            del os.environ["MODEL_VARIANT"]

    def tearDown(self):
        ModelLoader.unload()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.old_override is not None:
            os.environ["ADAPTER_OVERRIDE_PATH"] = self.old_override
        elif "ADAPTER_OVERRIDE_PATH" in os.environ:
            del os.environ["ADAPTER_OVERRIDE_PATH"]
        if self.old_variant is not None:
            os.environ["MODEL_VARIANT"] = self.old_variant
        elif "MODEL_VARIANT" in os.environ:
            del os.environ["MODEL_VARIANT"]

    def test_missing_adapter_directory_raises_filenotfound(self):
        """1. Verify missing adapter directory raises FileNotFoundError without silent fallback."""
        nonexistent = Path(self.temp_dir) / "does_not_exist"
        os.environ["ADAPTER_OVERRIDE_PATH"] = str(nonexistent)

        with self.assertRaises(FileNotFoundError) as ctx:
            ModelLoader.load_model("stage5_v1")

        self.assertIn("Adapter directory for variant 'stage5_v1' does not exist", str(ctx.exception))
        self.assertIn("Silent fallback to base model is disabled", str(ctx.exception))
        self.assertFalse(ModelLoader.is_loaded())

    def test_missing_adapter_config_raises_filenotfound(self):
        """2. Verify missing adapter_config.json raises FileNotFoundError."""
        adapter_dir = Path(self.temp_dir) / "test_adapter"
        adapter_dir.mkdir(parents=True)
        os.environ["ADAPTER_OVERRIDE_PATH"] = str(adapter_dir)

        with self.assertRaises(FileNotFoundError) as ctx:
            ModelLoader.load_model("stage5_v1")

        self.assertIn("Adapter config missing", str(ctx.exception))
        self.assertFalse(ModelLoader.is_loaded())

    def test_corrupt_adapter_config_raises_valueerror(self):
        """3. Verify invalid JSON in adapter_config.json raises ValueError."""
        adapter_dir = Path(self.temp_dir) / "test_adapter"
        adapter_dir.mkdir(parents=True)
        with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            f.write("{invalid_json: true")
        os.environ["ADAPTER_OVERRIDE_PATH"] = str(adapter_dir)

        with self.assertRaises(ValueError) as ctx:
            ModelLoader.load_model("stage5_v1")

        self.assertIn("Invalid adapter_config.json", str(ctx.exception))
        self.assertFalse(ModelLoader.is_loaded())

    def test_missing_adapter_weights_raises_filenotfound(self):
        """4. Verify missing binary weights raises FileNotFoundError (config-only rejected)."""
        adapter_dir = Path(self.temp_dir) / "test_adapter"
        adapter_dir.mkdir(parents=True)
        with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump({"peft_type": "LORA", "r": 16}, f)
        os.environ["ADAPTER_OVERRIDE_PATH"] = str(adapter_dir)

        with self.assertRaises(FileNotFoundError) as ctx:
            ModelLoader.load_model("stage5_v1")

        self.assertIn("Adapter weights not found for variant 'stage5_v1'", str(ctx.exception))
        self.assertIn("Config-only adapters are not loadable", str(ctx.exception))
        self.assertFalse(ModelLoader.is_loaded())

    def test_empty_adapter_weights_raises_filenotfound(self):
        """5. Verify 0-byte weight binary is rejected as invalid."""
        adapter_dir = Path(self.temp_dir) / "test_adapter"
        adapter_dir.mkdir(parents=True)
        with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump({"peft_type": "LORA", "r": 16}, f)
        # Create empty 0-byte file
        (adapter_dir / "adapter_model.safetensors").touch()
        self.assertEqual((adapter_dir / "adapter_model.safetensors").stat().st_size, 0)
        os.environ["ADAPTER_OVERRIDE_PATH"] = str(adapter_dir)

        with self.assertRaises(FileNotFoundError) as ctx:
            ModelLoader.load_model("stage5_v1")

        self.assertIn("Expected non-empty 'adapter_model.safetensors'", str(ctx.exception))
        self.assertFalse(ModelLoader.is_loaded())

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("peft.PeftModel.from_pretrained")
    def test_failed_peft_loading_raises_runtimeerror(self, mock_peft, mock_base_model, mock_tok):
        """6. Verify exception inside PeftModel.from_pretrained raises RuntimeError (no silent fallback)."""
        adapter_dir = Path(self.temp_dir) / "test_adapter"
        adapter_dir.mkdir(parents=True)
        with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump({"peft_type": "LORA", "r": 16}, f)
        with open(adapter_dir / "adapter_model.safetensors", "wb") as f:
            f.write(b"dummy_weights_1234567890")
        os.environ["ADAPTER_OVERRIDE_PATH"] = str(adapter_dir)

        mock_tok.return_value = MagicMock()
        mock_base_model.return_value = MagicMock()
        mock_peft.side_effect = Exception("Weight shape mismatch in q_proj")

        with self.assertRaises(RuntimeError) as ctx:
            ModelLoader.load_model("stage5_v1", force_cpu=True)

        self.assertIn("Failed to load LoRA adapter for variant 'stage5_v1'", str(ctx.exception))
        self.assertIn("Weight shape mismatch", str(ctx.exception))
        self.assertFalse(ModelLoader.is_loaded())

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("peft.PeftModel.from_pretrained")
    def test_successful_adapter_loading_verifies_metadata(self, mock_peft, mock_base_model, mock_tok):
        """7. Verify successful adapter loading updates metadata and marks adapter_loaded=True."""
        adapter_dir = Path(self.temp_dir) / "test_adapter"
        adapter_dir.mkdir(parents=True)
        with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump({"peft_type": "LORA", "r": 16}, f)
        with open(adapter_dir / "adapter_model.safetensors", "wb") as f:
            f.write(b"valid_model_weights_tensor_content")
        os.environ["ADAPTER_OVERRIDE_PATH"] = str(adapter_dir)

        dummy_tok = MagicMock()
        dummy_base = MagicMock()
        from peft import PeftModel
        dummy_peft = MagicMock(spec=PeftModel)

        mock_tok.return_value = dummy_tok
        mock_base_model.return_value = dummy_base
        mock_peft.return_value = dummy_peft

        m, t = ModelLoader.load_model("stage5_v1", force_cpu=True)
        self.assertIs(m, dummy_peft)
        self.assertIs(t, dummy_tok)
        self.assertTrue(ModelLoader.is_loaded())

        info = ModelLoader.get_info()
        self.assertEqual(info["model_id"], "stage5_v1")
        self.assertEqual(info["base_model"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(info["adapter_path"], str(adapter_dir))
        self.assertTrue(info["adapter_loaded"])
        self.assertFalse(info["is_mock"])

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("peft.PeftModel.from_pretrained")
    def test_successful_candidate_3b_adapter_loading(self, mock_peft, mock_base_model, mock_tok):
        """8. Verify candidate 3B adapter loads with correct 3B base model and adapter metadata."""
        adapter_dir = Path(self.temp_dir) / "stage7a_qwen2.5_3b"
        adapter_dir.mkdir(parents=True)
        with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump({"peft_type": "LORA", "r": 16, "base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}, f)
        with open(adapter_dir / "adapter_model.safetensors", "wb") as f:
            f.write(b"valid_3b_candidate_weights")
        os.environ["ADAPTER_OVERRIDE_PATH"] = str(adapter_dir)

        dummy_tok = MagicMock()
        dummy_base = MagicMock()
        from peft import PeftModel
        dummy_peft = MagicMock(spec=PeftModel)

        mock_tok.return_value = dummy_tok
        mock_base_model.return_value = dummy_base
        mock_peft.return_value = dummy_peft

        m, t = ModelLoader.load_model("stage7a_qwen2.5_3b", force_cpu=True)
        self.assertTrue(ModelLoader.is_loaded())

        info = ModelLoader.get_info()
        self.assertEqual(info["model_id"], "stage7a_qwen2.5_3b")
        self.assertEqual(info["base_model"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertTrue(info["adapter_loaded"])
        self.assertFalse(info["is_mock"])

    def test_invalid_variant_rejected(self):
        """9. Verify non-existent model variants raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ModelLoader.load_registry("invalid_variant_999_xyz")

        with self.assertRaises(FileNotFoundError):
            ModelLoader.load_model("invalid_variant_999_xyz")

    def test_baseline_default_invariance(self):
        """10. Verify default model selection remains stage5_v1."""
        meta = ModelLoader.load_registry()
        self.assertEqual(meta["model_id"], "stage5_v1")
        self.assertEqual(meta["base_model"], "Qwen/Qwen2.5-1.5B-Instruct")

        gen = PersonaGenerator()
        self.assertEqual(gen.model_id, "stage5_v1")
        self.assertEqual(gen.base_model_name, "Qwen/Qwen2.5-1.5B-Instruct")

    def test_isolated_mock_mode_behavior(self):
        """11. Verify intentional mock mode is isolated (is_mock=True, adapter_loaded=False)."""
        mock_m = MagicMock()
        mock_t = MagicMock()
        ModelLoader.load_model(model_id="stage7a_qwen2.5_3b", mock_model=mock_m, mock_tokenizer=mock_t)

        info = ModelLoader.get_info()
        self.assertTrue(info["loaded"])
        self.assertEqual(info["model_id"], "stage7a_qwen2.5_3b")
        self.assertTrue(info["is_mock"])
        # Mock mode MUST NOT claim real adapter is loaded
        self.assertFalse(info["adapter_loaded"])


if __name__ == "__main__":
    unittest.main()
