"""
Unit tests for Persona Inference Module & FastAPI Service.
Uses mocks to avoid downloading model weights during unit test execution.
"""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from ml.service.main import app
from ml.service.schemas import (
    ChatMessage,
    GenerateRequest,
    GenerationSettings,
    HealthResponse,
    ReadyResponse,
)
from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator
from ml.src.inference.prompt_builder import PromptBuilder, SYSTEM_PROMPT


class TestInferenceRegistryAndSchemas(unittest.TestCase):
    """Test registry loading and schema validation."""

    def setUp(self):
        self.ml_root = Path(__file__).resolve().parents[1]

    def test_registry_metadata_loads_correctly(self):
        """1. Verify registry metadata loads correctly and has all required fields."""
        metadata = ModelLoader.load_registry("stage5_v1")
        self.assertEqual(metadata["model_id"], "stage5_v1")
        self.assertEqual(metadata["status"], "baseline_persona_model")
        self.assertEqual(metadata["base_model"], "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(metadata["adapter_path"], "ml/models/adapters/stage5_v1")
        self.assertEqual(metadata["best_epoch"], 2)
        self.assertEqual(metadata["best_validation_loss"], 1.7645)
        self.assertEqual(metadata["training_method"], "QLoRA")
        self.assertEqual(metadata["quantization"], "NF4 4-bit")
        self.assertEqual(metadata["lora_rank"], 16)
        self.assertEqual(metadata["lora_alpha"], 32)
        self.assertEqual(metadata["lora_dropout"], 0.05)
        self.assertEqual(metadata["target_modules"], ["q_proj", "k_proj", "v_proj", "o_proj"])
        self.assertEqual(metadata["learning_rate"], 0.0002)
        self.assertEqual(metadata["epochs"], 3)
        self.assertEqual(metadata["effective_batch_size"], 8)
        self.assertEqual(metadata["max_sequence_length"], 512)
        self.assertEqual(metadata["evaluation_status"], "completed")

    def test_request_schema_validation(self):
        """2. Verify valid GenerateRequest schema instantiation."""
        req = GenerateRequest(
            messages=[
                ChatMessage(role="user", content="kal college aayega?"),
                ChatMessage(role="assistant", content="nhi bhai"),
                ChatMessage(role="user", content="kyu?"),
            ],
            generation=GenerationSettings(
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=48,
                repetition_penalty=1.1,
            )
        )
        self.assertEqual(len(req.messages), 3)
        self.assertEqual(req.generation.temperature, 0.8)
        self.assertEqual(req.generation.max_new_tokens, 48)

    def test_empty_message_rejection(self):
        """3. Verify rejection of empty messages list and empty content strings."""
        # Empty messages list
        with self.assertRaises(Exception):
            GenerateRequest(messages=[])

        # Empty content string
        with self.assertRaises(Exception):
            ChatMessage(role="user", content="")

        # Whitespace-only content string
        with self.assertRaises(Exception):
            ChatMessage(role="user", content="    ")

    def test_invalid_role_rejection(self):
        """4. Verify rejection of unsupported roles."""
        for invalid_role in ["bot", "admin", "moderator", "agent"]:
            with self.assertRaises(Exception):
                ChatMessage(role=invalid_role, content="hello")

    def test_prompt_construction(self):
        """7. Verify prompt construction logic and system prompt injection."""
        messages = [
            {"role": "user", "content": "Aaja"},
        ]
        prepared = PromptBuilder.prepare_messages(messages)
        self.assertEqual(len(prepared), 2)
        self.assertEqual(prepared[0]["role"], "system")
        self.assertEqual(prepared[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(prepared[1]["role"], "user")
        self.assertEqual(prepared[1]["content"], "Aaja")

        # Fallback text representation contains ChatML tokens
        prompt_text = PromptBuilder.build_prompt_text(messages)
        self.assertIn("<|im_start|>system", prompt_text)
        self.assertIn("<|im_start|>user\nAaja<|im_end|>", prompt_text)
        self.assertTrue(prompt_text.endswith("<|im_start|>assistant\n"))

    def test_generation_config_defaults_and_validation(self):
        """Verify generation config sensible defaults and boundary checks."""
        cfg = GenerationConfig()
        self.assertEqual(cfg.temperature, 0.7)
        self.assertEqual(cfg.top_p, 0.9)
        self.assertEqual(cfg.max_new_tokens, 64)
        self.assertEqual(cfg.repetition_penalty, 1.05)
        self.assertTrue(cfg.do_sample)

        # Greedy mode when temperature is 0
        cfg_greedy = GenerationConfig(temperature=0.0)
        self.assertFalse(cfg_greedy.do_sample)

        # Invalid bounds
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=1.5)
        with self.assertRaises(ValueError):
            GenerationConfig(max_new_tokens=0)


class TestInferenceApiEndpoints(unittest.TestCase):
    """Test FastAPI HTTP endpoints using starlette TestClient."""

    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        ModelLoader.unload()

    def tearDown(self):
        ModelLoader.unload()

    def test_health_endpoint(self):
        """5. Verify GET /health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model"], "stage5_v1")

    def test_readiness_endpoint(self):
        """6. Verify GET /ready endpoint reflection of model load state."""
        # Before loading -> ready is False
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ready"])
        self.assertEqual(response.json()["model"], "stage5_v1")

        # Inject mock model -> ready becomes True
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        ModelLoader.load_model(mock_model=mock_model, mock_tokenizer=mock_tokenizer)

        response_loaded = self.client.get("/ready")
        self.assertEqual(response_loaded.status_code, 200)
        self.assertTrue(response_loaded.json()["ready"])

    def test_generate_endpoint_validation_errors(self):
        """Verify POST /generate returns HTTP 400 on invalid requests."""
        # Empty payload
        resp_empty = self.client.post("/generate", json={})
        self.assertEqual(resp_empty.status_code, 400)
        self.assertIn("error", resp_empty.json())

        # Empty messages list
        resp_no_msgs = self.client.post("/generate", json={"messages": []})
        self.assertEqual(resp_no_msgs.status_code, 400)

        # Invalid role
        resp_bad_role = self.client.post("/generate", json={
            "messages": [{"role": "robot", "content": "hi"}]
        })
        self.assertEqual(resp_bad_role.status_code, 400)

        # Empty content
        resp_empty_content = self.client.post("/generate", json={
            "messages": [{"role": "user", "content": "  "}]
        })
        self.assertEqual(resp_empty_content.status_code, 400)

    def test_generation_interface_with_mock(self):
        """8. Verify POST /generate returns model completion with mock."""
        import torch

        # Setup mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "system prompt\nuser prompt\nassistant prompt"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 151645
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 20, 30]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        # mock.generate returns prompt tokens + 3 generated tokens
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        mock_model.generate.return_value = torch.tensor([[10, 20, 30, 40, 50, 60]])
        mock_tokenizer.decode.return_value = "aaja canteen chalte hai"

        ModelLoader.load_model(mock_model=mock_model, mock_tokenizer=mock_tokenizer)

        payload = {
            "messages": [
                {"role": "user", "content": "free hai kya abhi?"},
                {"role": "assistant", "content": "assignment submit kar raha tha"},
                {"role": "user", "content": "Khelega?"}
            ],
            "generation": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 32,
                "repetition_penalty": 1.05
            }
        }

        response = self.client.post("/generate", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response"], "aaja canteen chalte hai")
        self.assertEqual(data["model"], "stage5_v1")
        # Ensure generate was invoked
        self.assertTrue(mock_model.generate.called)


if __name__ == "__main__":
    unittest.main()
