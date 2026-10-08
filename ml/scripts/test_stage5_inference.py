#!/usr/bin/env python3
"""
Persona Engine — Stage 5 Inference Smoke Test
Tests end-to-end inference using the Stage 5 baseline persona model (Qwen2.5-1.5B + LoRA adapter).

Executes the 4 required benchmark test scenarios:
Scenario A: Standalone ambiguous prompt "Aaja"
Scenario B: Rich dialogue context leading to "Aaja" (canteen meeting)
Scenario C: Rich dialogue context leading to "Khelega?" (finishing assignment)
Scenario D: Rich dialogue context leading to "Game aaja" (after dinner)
"""

import argparse
import logging
from pathlib import Path
import sys
import time

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root and ml root in sys.path
ml_root = Path(__file__).resolve().parents[1]
repo_root = ml_root.parent
for p in [str(repo_root), str(ml_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage5SmokeTest")


TEST_SCENARIOS = [
    {
        "name": "Scenario A: Standalone Ambiguous Prompt ('Aaja')",
        "messages": [
            {"role": "user", "content": "Aaja"}
        ],
        "description": "Tests default unconditioned brevity on ambiguous prompt."
    },
    {
        "name": "Scenario B: Contextual Canteen Prompt ('Aaja')",
        "messages": [
            {"role": "user", "content": "canteen me milte hai?"},
            {"role": "assistant", "content": "5 min me pohochta hu"},
            {"role": "user", "content": "Aaja"}
        ],
        "description": "Tests context sensitivity to earlier canteen arrival commitment."
    },
    {
        "name": "Scenario C: Contextual Activity Prompt ('Khelega?')",
        "messages": [
            {"role": "user", "content": "free hai kya abhi?"},
            {"role": "assistant", "content": "assignment submit kar raha tha"},
            {"role": "user", "content": "Khelega?"}
        ],
        "description": "Tests context sensitivity to pending assignment completion status."
    },
    {
        "name": "Scenario D: Contextual Gaming Prompt ('Game aaja')",
        "messages": [
            {"role": "user", "content": "dinner kar liya?"},
            {"role": "assistant", "content": "haa abhi kiya"},
            {"role": "user", "content": "Game aaja"}
        ],
        "description": "Tests context sensitivity to post-dinner gaming readiness."
    }
]


def run_smoke_test(use_mock: bool = False, force_cpu: bool = False):
    print("=" * 75)
    print("STAGE 5 — PERSONA INFERENCE SERVICE SMOKE TEST")
    print("=" * 75)

    if use_mock:
        print("[MODE] Using simulated mock model for lightweight CI/smoke testing.")
        import torch
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 151645
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 20, 30]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        mock_responses = [
            "Aaya",
            "Canteen pe hi hu aaja",
            "Assignment bas submit kar raha hu fir aata hu",
            "Haa login kar raha hu aaja"
        ]
        resp_iter = iter(mock_responses)

        def mock_generate(*args, **kwargs):
            return torch.tensor([[10, 20, 30, 40, 50]])

        def mock_decode(*args, **kwargs):
            return next(resp_iter, "Sahi hai")

        mock_model = MagicMock()
        mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])
        mock_model.generate = mock_generate
        mock_tokenizer.decode = mock_decode

        ModelLoader.load_model(mock_model=mock_model, mock_tokenizer=mock_tokenizer)
    else:
        print("[MODE] Loading actual Stage 5 model from disk / registry...")
        ModelLoader.load_model(model_id="stage5_v1", force_cpu=force_cpu)

    info = ModelLoader.get_info()
    print(f"Serving Model : {info['model_id']}")
    print(f"Base Model    : {info['base_model']}")
    print(f"Device        : {info['device']}")
    print(f"Quantization  : {info['quantization']}")
    print(f"Load Time     : {info['load_duration_seconds']}s")
    print("-" * 75)

    generator = PersonaGenerator(model_id="stage5_v1")
    config = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=48)

    results = []
    for idx, sc in enumerate(TEST_SCENARIOS, 1):
        print(f"\n[{idx}/4] {sc['name']}")
        print(f"  Description: {sc['description']}")
        print("  Input Context:")
        for turn in sc["messages"]:
            print(f"    - {turn['role'].capitalize()}: {turn['content']}")

        t0 = time.time()
        resp = generator.generate(sc["messages"], generation_config=config, request_id=f"smoke_{idx}")
        latency = (time.time() - t0) * 1000

        print(f"  Generated Persona Response: \"{resp}\"")
        print(f"  Generation Latency        : {latency:.1f}ms")

        results.append({
            "scenario": sc["name"],
            "messages": sc["messages"],
            "response": resp,
            "latency_ms": latency
        })

    print("\n" + "=" * 75)
    print("ALL 4 SMOKE TEST SCENARIOS EXECUTED SUCCESSFULLY")
    print("=" * 75)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Stage 5 Inference")
    parser.add_argument("--mock", action="store_true", help="Run with mock model (for lightweight testing)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    run_smoke_test(use_mock=args.mock, force_cpu=args.cpu)
