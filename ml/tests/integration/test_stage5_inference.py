"""
Integration test for Stage 5 Persona Inference.
Executes against actual model weights if CUDA is available; skips gracefully on CPU/CI.
"""

import unittest
from pathlib import Path
import sys

# Ensure repository root and ml root in sys.path
ml_root = Path(__file__).resolve().parents[2]
repo_root = ml_root.parent
for p in [str(repo_root), str(ml_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator


class TestStage5InferenceIntegration(unittest.TestCase):
    """Optional integration tests requiring GPU / loaded weights."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU not available; skipping heavy model integration test")
    def test_live_stage5_inference_gpu(self):
        """Live generation test on GPU with actual Qwen2.5-1.5B weights."""
        generator = PersonaGenerator(model_id="stage5_v1")
        messages = [
            {"role": "user", "content": "canteen me milte hai?"},
            {"role": "assistant", "content": "5 min me pohochta hu"},
            {"role": "user", "content": "Aaja"}
        ]
        cfg = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=32)
        response = generator.generate(messages, generation_config=cfg)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
