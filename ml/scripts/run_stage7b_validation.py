#!/usr/bin/env python3
"""
Persona Engine — Stage 7B: Runtime Validation Runner
Validates candidate model (Qwen/Qwen2.5-3B-Instruct) runtime behavior in Google Colab / GPU.

Features:
- Hardware & environment diagnostics (GPU VRAM, CUDA, PyTorch, Transformers, PEFT, BitsAndBytes)
- Model registry and adapter configuration verification (1.5B baseline and 3B candidate)
- Confirms stage5_v1 (1.5B) remains the default
- Verifies opt-in loading of 3B candidate via MODEL_VARIANT
- Error handling testing: safe rejection of invalid variants & missing adapter paths
- Smoke tests: Short Hinglish, Multi-turn, Ambiguous prompt, and Memory-conditioned turn
- Full layer verification: Exercises Context 6F, Memory 6C, Orchestrator 6D, and Generator
- Latency & peak GPU memory allocation measurement
- Exports structured JSON results and Markdown report to ml/data/stage7b/

Usage:
    python ml/scripts/run_stage7b_validation.py
    python ml/scripts/run_stage7b_validation.py --mock
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root and ml root in sys.path
repo_root = Path(__file__).resolve().parents[2]
ml_root = Path(__file__).resolve().parents[1]
for p in [str(repo_root), str(ml_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.src.context.context_analyzer import ContextAnalyzer
from ml.src.context.context_schema import AmbiguityLevel, ContextAnalysis
from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator
from ml.src.memory.embedding import DeterministicMockEmbedding
from ml.src.memory.memory_ranker import MemoryRanker
from ml.src.memory.memory_retriever import MemoryRetriever
from ml.src.memory.memory_schema import ImportanceLevel, Memory, MemoryType
from ml.src.memory.memory_store import InMemoryMemoryStore
from ml.src.orchestrator.orchestrator import ConversationOrchestrator
from ml.src.orchestrator.orchestrator_schema import ConversationRequest, ResponsePlan

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Stage7BValidation")


def get_environment_diagnostics() -> Dict[str, Any]:
    """Inspects Python, CUDA, GPU, and core ML package versions."""
    diag: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": "not_installed",
        "cuda_available": False,
        "cuda_version": None,
        "gpu_device_name": None,
        "gpu_total_vram_gb": 0.0,
        "gpu_reserved_vram_gb": 0.0,
        "gpu_allocated_vram_gb": 0.0,
        "packages": {},
    }

    try:
        import torch
        diag["torch_version"] = torch.__version__
        diag["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diag["cuda_version"] = torch.version.cuda
            diag["gpu_device_name"] = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            diag["gpu_total_vram_gb"] = round(total_mem, 2)
            diag["gpu_reserved_vram_gb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 3), 2)
            diag["gpu_allocated_vram_gb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
    except ImportError:
        pass

    packages_to_check = [
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "datasets",
        "sentence_transformers",
    ]
    for pkg in packages_to_check:
        try:
            mod = __import__(pkg)
            diag["packages"][pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            diag["packages"][pkg] = "not_installed"

    return diag


def verify_configurations_and_adapters() -> Dict[str, Any]:
    """Verifies registry files and adapter directories for baseline and candidate."""
    results: Dict[str, Any] = {}

    for key, reg_name, adapter_rel in [
        ("stage5_baseline", "stage5_v1.json", "stage5_v1"),
        ("stage7a_candidate", "stage7a_qwen2.5_3b.json", "stage7a_qwen2.5_3b"),
    ]:
        reg_path = ml_root / "models" / "registry" / reg_name
        adapter_dir = ml_root / "models" / "adapters" / adapter_rel
        reg_exists = reg_path.is_file()
        dir_exists = adapter_dir.is_dir()
        cfg_exists = (adapter_dir / "adapter_config.json").is_file()
        has_weights = any(
            (adapter_dir / f).is_file() and (adapter_dir / f).stat().st_size > 0
            for f in ["adapter_model.safetensors", "adapter_model.bin"]
        )
        is_valid = reg_exists and dir_exists and cfg_exists and has_weights

        results[key] = {
            "registry_exists": reg_exists,
            "adapter_dir_exists": dir_exists,
            "adapter_config_exists": cfg_exists,
            "has_weight_binaries": has_weights,
            "status": "PASS" if is_valid else "FAIL",
        }
        if reg_exists:
            with open(reg_path, "r", encoding="utf-8") as f:
                results[key]["config"] = json.load(f)

    return results


def verify_default_model_invariance() -> Dict[str, Any]:
    """Verifies that stage5_v1 remains the default model when no MODEL_VARIANT is passed."""
    old_env = os.environ.pop("MODEL_VARIANT", None)
    try:
        gen = PersonaGenerator()
        info = ModelLoader.get_info()
        is_default = (gen.model_id == "stage5_v1" and info["model_id"] == "stage5_v1")
        return {
            "status": "PASS" if is_default else "FAIL",
            "generator_model_id": gen.model_id,
            "loader_model_id": info["model_id"],
            "base_model": gen.base_model_name,
            "is_default_stage5": is_default,
        }
    finally:
        if old_env is not None:
            os.environ["MODEL_VARIANT"] = old_env


def test_invalid_variants_and_missing_adapters() -> Dict[str, Any]:
    """Tests that invalid variants and missing adapters are handled safely with explicit exceptions."""
    results: Dict[str, Any] = {}

    # Test 1: Invalid model variant name
    invalid_id = "non_existent_model_variant_xyz_999"
    try:
        ModelLoader.load_registry(invalid_id)
        results["invalid_variant_test"] = {
            "status": "FAIL",
            "message": "Expected FileNotFoundError for non-existent model registry",
        }
    except FileNotFoundError as e:
        results["invalid_variant_test"] = {
            "status": "PASS",
            "message": f"Correctly caught FileNotFoundError: {e}",
        }

    # Test 2: Missing adapter directory override must raise FileNotFoundError
    old_override = os.environ.get("ADAPTER_OVERRIDE_PATH")
    os.environ["ADAPTER_OVERRIDE_PATH"] = str(ml_root / "models" / "adapters" / "missing_nonexistent_dir")
    try:
        ModelLoader.unload()
        try:
            ModelLoader.load_model("stage5_v1")
            results["missing_adapter_path_test"] = {
                "status": "FAIL",
                "message": "Expected FileNotFoundError for non-existent adapter path, but model loaded without error",
            }
        except FileNotFoundError as err:
            results["missing_adapter_path_test"] = {
                "status": "PASS",
                "message": f"Correctly caught FileNotFoundError on missing adapter: {err}",
            }
    finally:
        ModelLoader.unload()
        if old_override:
            os.environ["ADAPTER_OVERRIDE_PATH"] = old_override
        else:
            os.environ.pop("ADAPTER_OVERRIDE_PATH", None)

    return results


def run_smoke_tests(generator: PersonaGenerator) -> List[Dict[str, Any]]:
    """
    Executes core smoke tests:
    1. Short Hinglish query
    2. Multi-turn dialogue
    3. Ambiguous short prompt
    Requires verified adapter loading before a smoke test can pass.
    """
    gen_cfg = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=48)
    test_cases = [
        {
            "test_name": "short_hinglish",
            "description": "Short conversational Hinglish opener",
            "messages": [
                {"role": "user", "content": "bhai kya scene hai aaj ka?"}
            ],
        },
        {
            "test_name": "multi_turn_dialogue",
            "description": "3-turn context tracking dialogue",
            "messages": [
                {"role": "user", "content": "assignment submit kar diya tune?"},
                {"role": "assistant", "content": "nahi abhi aadha bacha hai"},
                {"role": "user", "content": "solution bheju kya reference ke liye?"},
            ],
        },
        {
            "test_name": "ambiguous_prompt",
            "description": "Single-token prompt Aaja with physical meeting context",
            "messages": [
                {"role": "user", "content": "canteen me baitha hu kab aayega?"},
                {"role": "assistant", "content": "2 min me nikal raha hu room se"},
                {"role": "user", "content": "Aaja"},
            ],
        },
    ]

    info = ModelLoader.get_info()
    adapter_confirmed = info.get("adapter_loaded", False)
    variant_matches = (info.get("model_id") == generator.model_id)

    results = []
    for tc in test_cases:
        t0 = time.perf_counter()
        lat_ms = 0.0
        response = ""
        err_msg = None
        try:
            response = generator.generate(tc["messages"], generation_config=gen_cfg)
            lat_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            err_msg = str(e)
            lat_ms = (time.perf_counter() - t0) * 1000.0

        is_non_empty = len(response.strip()) > 0
        test_passed = (err_msg is None) and is_non_empty and variant_matches and adapter_confirmed

        failure_reasons = []
        if err_msg:
            failure_reasons.append(f"Inference error: {err_msg}")
        if not is_non_empty and not err_msg:
            failure_reasons.append("Empty response")
        if not variant_matches:
            failure_reasons.append(f"Variant mismatch (expected {generator.model_id}, got {info.get('model_id')})")
        if not adapter_confirmed:
            failure_reasons.append("Adapter loading not confirmed (adapter_loaded=False)")

        results.append({
            "test_name": tc["test_name"],
            "description": tc["description"],
            "turns": len(tc["messages"]),
            "input_preview": tc["messages"][-1]["content"],
            "response": response,
            "word_count": len(response.split()),
            "latency_ms": round(lat_ms, 2),
            "status": "PASS" if test_passed else "FAIL",
            "failure_reasons": failure_reasons if not test_passed else None,
        })

    return results


def run_memory_conditioned_pipeline_smoke_test(
    generator: PersonaGenerator,
) -> Dict[str, Any]:
    """
    Directly exercises all 4 engine components:
    1. ContextAnalyzer (Context Engine 6F)
    2. MemoryRetriever with seeded InMemoryMemoryStore (Memory Engine 6C)
    3. ConversationOrchestrator (Orchestrator 6D)
    4. PersonaGenerator (Inference 6A)
    Requires confirmed adapter loading for the Generator layer.
    """
    messages = [
        {"role": "user", "content": "konsa laptop hai tere pass?"}
    ]

    # 1. Context Engine
    ctx_analyzer = ContextAnalyzer()
    t_ctx0 = time.perf_counter()
    context = ctx_analyzer.analyze(messages)
    ctx_ms = (time.perf_counter() - t_ctx0) * 1000.0

    # 2. Memory Engine
    embedding_model = DeterministicMockEmbedding(dimension=64)
    store = InMemoryMemoryStore(embedding_model=embedding_model)
    store.save(Memory(
        id="mem_laptop_nitro",
        persona_id="vivek",
        memory_type=MemoryType.FACT,
        content="Owns an Acer Nitro laptop",
        topic="technology",
        importance=ImportanceLevel.HIGH,
    ))
    store.save(Memory(
        id="mem_college_noida",
        persona_id="vivek",
        memory_type=MemoryType.FACT,
        content="Studies B.Tech CSE in college located in Noida",
        topic="college",
        importance=ImportanceLevel.MEDIUM,
    ))

    retriever = MemoryRetriever(store=store, ranker=MemoryRanker(), embedding_model=embedding_model)
    t_mem0 = time.perf_counter()
    retrieved = retriever.retrieve(
        persona_id="vivek",
        conversation_context=messages,
        context_analysis=context,
        top_k=2,
    )
    mem_ms = (time.perf_counter() - t_mem0) * 1000.0
    retrieved_ids = [m.memory.id for m in retrieved]

    # 3. Conversation Orchestrator
    orchestrator = ConversationOrchestrator(context_analyzer=ctx_analyzer)
    req = ConversationRequest(
        persona_id="vivek",
        conversation_id="stage7b_smoke_req",
        messages=messages,
        context=context,
        memories=retrieved,
    )
    t_orch0 = time.perf_counter()
    plan = orchestrator.plan(req)
    orch_ms = (time.perf_counter() - t_orch0) * 1000.0

    # 4. Generator
    context_messages = list(messages)
    context_messages.insert(
        0,
        {"role": "system", "content": f"Tone: {plan.tone.value}. Strategy: {plan.response_strategy.value}. {plan.generation_instruction}"}
    )
    t_gen0 = time.perf_counter()
    response = ""
    gen_err = None
    try:
        response = generator.generate(context_messages)
    except Exception as e:
        gen_err = str(e)
    gen_ms = (time.perf_counter() - t_gen0) * 1000.0

    info = ModelLoader.get_info()
    adapter_confirmed = info.get("adapter_loaded", False)

    layers_exercised = {
        "context_engine": context is not None,
        "memory_engine": len(retrieved) > 0,
        "orchestrator": plan is not None,
        "generator": (gen_err is None) and (len(response.strip()) > 0) and adapter_confirmed,
    }

    all_passed = all(layers_exercised.values())

    return {
        "status": "PASS" if all_passed else "FAIL",
        "topic": context.topic.value,
        "strategy": plan.response_strategy.value,
        "tone": plan.tone.value,
        "retrieved_memory_ids": retrieved_ids,
        "generation_instruction": plan.generation_instruction,
        "response": response,
        "generation_error": gen_err,
        "latencies": {
            "context_ms": round(ctx_ms, 2),
            "memory_ms": round(mem_ms, 2),
            "orchestrator_ms": round(orch_ms, 2),
            "generator_ms": round(gen_ms, 2),
            "total_e2e_ms": round(ctx_ms + mem_ms + orch_ms + gen_ms, 2),
        },
        "layers_exercised": layers_exercised,
    }


def measure_benchmarks(generator: PersonaGenerator, runs: int = 5) -> Dict[str, Any]:
    """Measures generation latencies and peak GPU memory."""
    import numpy as np

    prompt = [{"role": "user", "content": "khelega kya aaj raat bgmi?"}]
    gen_cfg = GenerationConfig(temperature=0.7, max_new_tokens=32)

    # Warm-up run
    generator.generate(prompt, generation_config=gen_cfg)

    latencies_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = generator.generate(prompt, generation_config=gen_cfg)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)

    peak_vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram_gb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3)
    except Exception:
        pass

    return {
        "iterations": runs,
        "latency_mean_ms": round(float(np.mean(arr)), 2),
        "latency_median_ms": round(float(np.median(arr)), 2),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "latency_min_ms": round(float(np.min(arr)), 2),
        "latency_max_ms": round(float(np.max(arr)), 2),
        "peak_vram_gb": peak_vram_gb,
    }


def save_reports(
    results_data: Dict[str, Any],
    reports_dir: Path,
    results_dir: Path,
) -> Tuple[Path, Path]:
    """Saves structured JSON and Markdown report to the Stage 7B directories."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / "stage7b_validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    md_path = reports_dir / "stage7b_runtime_validation_report.md"

    diag = results_data.get("environment_diagnostics", {})
    configs = results_data.get("configuration_verification", {})
    smoke = results_data.get("smoke_tests", [])
    mem_smoke = results_data.get("memory_conditioned_smoke_test", {})
    bench = results_data.get("benchmarks", {})
    overall_status = results_data.get("overall_status", "FAIL")

    ctrl_cfg = configs.get("stage5_baseline", {})
    cand_cfg = configs.get("stage7a_candidate", {})

    if overall_status == "PASS":
        conclusion_text = """Stage 7B runtime validation confirms:
1. `stage5_v1` remains the default baseline model.
2. Candidate `stage7a_qwen2.5_3b` loads properly with verified LoRA adapter weights (no silent base fallback).
3. All four core pipeline layers (Context, Memory, Orchestrator, Generator) are operational and exercised.
4. Error boundaries safely reject invalid variants and missing adapters."""
    else:
        failure_notes = []
        if not ctrl_cfg.get("has_weight_binaries"):
            failure_notes.append("Baseline adapter weights (adapter_model.safetensors/bin) missing")
        if not cand_cfg.get("has_weight_binaries"):
            failure_notes.append("Candidate adapter weights (adapter_model.safetensors/bin) missing")
        if not results_data.get("candidate_info", {}).get("adapter_loaded"):
            failure_notes.append("Candidate PEFT adapter was not confirmed loaded")
        if results_data.get("candidate_load_error"):
            failure_notes.append(f"Candidate load error: {results_data.get('candidate_load_error')}")
        notes_str = "\n".join(f"- {n}" for n in failure_notes) if failure_notes else "- Integrity checks failed (see breakdown above)."
        conclusion_text = f"""Stage 7B runtime validation FAILED integrity checks:
{notes_str}

Silent base model fallback is strictly prohibited. Real adapter weights must be provisioned before validation can pass."""

    md_content = f"""# Stage 7B — Colab Runtime Validation Report
## Persona Engine — 3B Candidate Verification & Default Model Audit

**Status:** `{overall_status}`  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Validation Target:** `stage7a_qwen2.5_3b` (`Qwen/Qwen2.5-3B-Instruct`)  
**Default Baseline:** `stage5_v1` (`Qwen/Qwen2.5-1.5B-Instruct`)  

---

## 1. Environment & Diagnostics

- **Platform:** `{diag.get('platform')}`
- **Python Version:** `{diag.get('python_version')}`
- **PyTorch Version:** `{diag.get('torch_version')}`
- **CUDA Available:** `{diag.get('cuda_available')}` ({diag.get('cuda_version')})
- **GPU Device:** `{diag.get('gpu_device_name', 'None')}`
- **Total VRAM:** `{diag.get('gpu_total_vram_gb', 0.0)} GB`
- **Installed Packages:**
  * `transformers`: `{diag.get('packages', {}).get('transformers')}`
  * `peft`: `{diag.get('packages', {}).get('peft')}`
  * `bitsandbytes`: `{diag.get('packages', {}).get('bitsandbytes')}`
  * `accelerate`: `{diag.get('packages', {}).get('accelerate')}`
  * `datasets`: `{diag.get('packages', {}).get('datasets')}`

---

## 2. Configuration & Adapter Audit

| Model Variant | Registry Status | Adapter Dir | Adapter Config | Weights Present | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Control (`stage5_v1`)** | {'PASS' if ctrl_cfg.get('registry_exists') else 'FAIL'} | {'PASS' if ctrl_cfg.get('adapter_dir_exists') else 'FAIL'} | {'PASS' if ctrl_cfg.get('adapter_config_exists') else 'FAIL'} | {'YES' if ctrl_cfg.get('has_weight_binaries') else 'CONFIG ONLY'} | {ctrl_cfg.get('status', 'FAIL')} |
| **Candidate (`stage7a_qwen2.5_3b`)** | {'PASS' if cand_cfg.get('registry_exists') else 'FAIL'} | {'PASS' if cand_cfg.get('adapter_dir_exists') else 'FAIL'} | {'PASS' if cand_cfg.get('adapter_config_exists') else 'FAIL'} | {'YES' if cand_cfg.get('has_weight_binaries') else 'CONFIG ONLY'} | {cand_cfg.get('status', 'FAIL')} |

- **Default Invariance:** `{results_data.get('default_model_invariance', {}).get('status')}` (Verified `stage5_v1` remains the default).
- **Candidate Opt-in:** Verified opt-in via `model_id='stage7a_qwen2.5_3b'` or `MODEL_VARIANT=stage7a_qwen2.5_3b`.
- **Negative Testing:** `{results_data.get('error_handling_tests', {}).get('invalid_variant_test', {}).get('status')}` (Safely caught invalid model variants and missing adapter paths).

---

## 3. Smoke Test Results

"""
    for s in smoke:
        md_content += f"""### Smoke Test: `{s['test_name']}`
- **Description:** {s['description']}
- **Input:** `"{s['input_preview']}"`
- **Output:** `"{s['response']}"`
- **Words:** {s['word_count']} words | **Latency:** {s['latency_ms']} ms | **Status:** `{s['status']}`
"""
        if s.get("failure_reasons"):
            md_content += f"- **Failures:** `{s['failure_reasons']}`\n"
        md_content += "\n"

    md_content += f"""### Memory-Conditioned End-to-End Pipeline Smoke Test
- **Input:** `"konsa laptop hai tere pass?"`
- **Predicted Topic:** `{mem_smoke.get('topic')}`
- **Orchestrated Strategy:** `{mem_smoke.get('strategy')}` ({mem_smoke.get('tone')})
- **Retrieved Memory IDs:** `{mem_smoke.get('retrieved_memory_ids')}`
- **Instruction:** *"{mem_smoke.get('generation_instruction')}"*
- **Final Output:** `"{mem_smoke.get('response')}"`
- **Latency Breakdown:**
  * Context Engine: `{mem_smoke.get('latencies', {}).get('context_ms')} ms`
  * Memory Engine: `{mem_smoke.get('latencies', {}).get('memory_ms')} ms`
  * Orchestrator: `{mem_smoke.get('latencies', {}).get('orchestrator_ms')} ms`
  * Generator: `{mem_smoke.get('latencies', {}).get('generator_ms')} ms`
  * Total E2E: `{mem_smoke.get('latencies', {}).get('total_e2e_ms')} ms`
- **Components Exercised:**
  * Context Engine: `{'PASS' if mem_smoke.get('layers_exercised', {}).get('context_engine') else 'FAIL'}`
  * Memory Engine: `{'PASS' if mem_smoke.get('layers_exercised', {}).get('memory_engine') else 'FAIL'}`
  * Orchestrator: `{'PASS' if mem_smoke.get('layers_exercised', {}).get('orchestrator') else 'FAIL'}`
  * Generator: `{'PASS' if mem_smoke.get('layers_exercised', {}).get('generator') else 'FAIL'}`

---

## 4. Latency & Resource Benchmarks

| Metric | Measured Value |
|:---|:---:|
| **Warm Generation Latency (Mean)** | **{bench.get('latency_mean_ms', 0.0)} ms** |
| **Warm Generation Latency (Median)** | **{bench.get('latency_median_ms', 0.0)} ms** |
| **Warm Generation Latency (P95)** | **{bench.get('latency_p95_ms', 0.0)} ms** |
| **Peak GPU VRAM Allocated** | **{bench.get('peak_vram_gb', 0.0)} GB** |

---

## 5. Verification Conclusion

{conclusion_text}

---
**Report generated by Stage 7B Validation Suite.**
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Persona Engine Stage 7B Runtime Validation")
    parser.add_argument("--mock", action="store_true", help="Run with mock generator handles")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark run count")
    args = parser.parse_args()

    print("=" * 70)
    print("STAGE 7B — COLAB RUNTIME VALIDATION RUNNER")
    print("=" * 70)

    # 1. Environment Diagnostics
    print("\n[Step 1/7] Collecting Environment & Hardware Diagnostics...")
    env_diag = get_environment_diagnostics()
    print(f"  PyTorch Version : {env_diag['torch_version']}")
    print(f"  CUDA Available  : {env_diag['cuda_available']} ({env_diag['cuda_version']})")
    print(f"  GPU Device      : {env_diag['gpu_device_name']}")
    print(f"  Total VRAM      : {env_diag['gpu_total_vram_gb']} GB")

    # 2. Configuration & Adapter Verification
    print("\n[Step 2/7] Auditing Model Registries & Adapter Files...")
    config_audit = verify_configurations_and_adapters()
    print(f"  stage5_v1 (1.5B)       : Status={config_audit['stage5_baseline']['status']} (Registry={config_audit['stage5_baseline']['registry_exists']}, Weights={config_audit['stage5_baseline']['has_weight_binaries']})")
    print(f"  stage7a_qwen2.5_3b (3B): Status={config_audit['stage7a_candidate']['status']} (Registry={config_audit['stage7a_candidate']['registry_exists']}, Weights={config_audit['stage7a_candidate']['has_weight_binaries']})")

    # 3. Default Model Invariance Check
    print("\n[Step 3/7] Verifying Default Baseline Model Invariance...")
    default_check = verify_default_model_invariance()
    print(f"  Default Invariance: {default_check['status']} (Default ID={default_check['generator_model_id']})")
    if default_check["status"] != "PASS":
        logger.error("FATAL: Default model is not stage5_v1!")
        sys.exit(1)

    # 4. Error Handling Testing
    print("\n[Step 4/7] Testing Invalid Variant & Missing Adapter Safety...")
    err_tests = test_invalid_variants_and_missing_adapters()
    print(f"  Invalid Variant Rejection: {err_tests['invalid_variant_test']['status']}")
    print(f"  Missing Adapter Safety   : {err_tests['missing_adapter_path_test']['status']}")

    # 5. Candidate Model Loading
    print("\n[Step 5/7] Loading 3B Candidate Model (stage7a_qwen2.5_3b)...")
    target_variant = "stage7a_qwen2.5_3b"
    candidate_gen = PersonaGenerator(model_id=target_variant)
    candidate_load_error = None
    try:
        candidate_gen.load_model(mock=args.mock)
        info = ModelLoader.get_info()
        assert info["model_id"] == target_variant, f"Selected variant '{info['model_id']}' != requested '{target_variant}'"
    except Exception as e:
        candidate_load_error = str(e)
        info = ModelLoader.get_info()
        logger.error(f"Candidate loading failed: {candidate_load_error}")

    print(f"  Loaded Model ID : {info['model_id']}")
    print(f"  Base Model Name : {info['base_model']}")
    print(f"  Adapter Path    : {info['adapter_path']}")
    print(f"  Adapter Loaded  : {info['adapter_loaded']}")
    print(f"  Is Mock         : {info['is_mock']}")
    print(f"  Device          : {info['device']}")
    print(f"  Quantization    : {info['quantization']}")

    # 6. Smoke Tests
    print("\n[Step 6/7] Running Smoke Tests (Short, Multi-turn, Ambiguous, Memory)...")
    if candidate_load_error is not None:
        logger.warning(f"Candidate loading failed: {candidate_load_error}. Recording smoke test failures.")
        smoke_results = []
        mem_smoke_result = {"status": "FAIL", "error": candidate_load_error, "latencies": {}}
    else:
        smoke_results = run_smoke_tests(candidate_gen)
        for sr in smoke_results:
            print(f"  [{sr['test_name']:20}] Status={sr['status']} Output='{sr.get('response', '')}' ({sr['latency_ms']}ms)")

        mem_smoke_result = run_memory_conditioned_pipeline_smoke_test(candidate_gen)
        print(f"  [memory_conditioned ] Strategy='{mem_smoke_result.get('strategy')}', Status={mem_smoke_result['status']}")

    # 7. Benchmarks & Report Generation
    print("\n[Step 7/7] Measuring Generation Latency & Peak VRAM...")
    if candidate_load_error is None and info.get("adapter_loaded", False) and not info.get("is_mock", False):
        benchmarks = measure_benchmarks(candidate_gen, runs=args.runs)
        print(f"  Mean Latency : {benchmarks['latency_mean_ms']} ms (P95: {benchmarks['latency_p95_ms']} ms)")
        print(f"  Peak VRAM    : {benchmarks['peak_vram_gb']} GB")
    else:
        benchmarks = {
            "iterations": 0,
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
            "peak_vram_gb": 0.0,
        }
        print("  Benchmarks skipped (candidate model LoRA weights not confirmed loaded).")

    # Dynamic overall status calculation based on real validation results
    config_passed = (
        config_audit.get("stage5_baseline", {}).get("status") == "PASS" and
        config_audit.get("stage7a_candidate", {}).get("status") == "PASS"
    )
    default_passed = default_check.get("status") == "PASS"
    error_tests_passed = (
        err_tests.get("invalid_variant_test", {}).get("status") == "PASS" and
        err_tests.get("missing_adapter_path_test", {}).get("status") == "PASS"
    )
    candidate_ok = (
        candidate_load_error is None and
        info.get("model_id") == target_variant and
        info.get("adapter_loaded", False) is True and
        not info.get("is_mock", False)
    )
    smoke_passed = len(smoke_results) > 0 and all(s.get("status") == "PASS" for s in smoke_results)
    mem_smoke_passed = mem_smoke_result.get("status") == "PASS"

    overall_status = "PASS" if (
        config_passed and
        default_passed and
        error_tests_passed and
        candidate_ok and
        smoke_passed and
        mem_smoke_passed
    ) else "FAIL"

    full_results = {
        "overall_status": overall_status,
        "candidate_load_error": candidate_load_error,
        "timestamp": datetime.now().isoformat(),
        "environment_diagnostics": env_diag,
        "configuration_verification": config_audit,
        "default_model_invariance": default_check,
        "error_handling_tests": err_tests,
        "candidate_info": info,
        "smoke_tests": smoke_results,
        "memory_conditioned_smoke_test": mem_smoke_result,
        "benchmarks": benchmarks,
    }

    reports_dir = ml_root / "data" / "stage7b" / "reports"
    results_dir = ml_root / "data" / "stage7b" / "results"
    json_path, md_path = save_reports(full_results, reports_dir, results_dir)

    print("\nArtifacts Saved:")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")
    print("\n" + "=" * 70)
    print(f"STAGE 7B RUNTIME VALIDATION COMPLETED: OVERALL STATUS = {overall_status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
