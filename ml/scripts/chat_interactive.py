#!/usr/bin/env python3
"""
Persona Engine — Interactive Chat & Progress Inspector
Interactive conversational interface to chat with the Vivek Persona Engine.

Features:
- Chat directly with the fine-tuned persona model or full 4-layer cognitive pipeline.
- Switch between Stage 5 baseline (stage5_v1) and Stage 7A candidate (stage7a_qwen2.5_3b).
- Optional full pipeline mode (--pipeline): shows live Context Analysis, Memory Retrieval,
  and Orchestrator planning alongside Vivek's generated response.
- Supports single-prompt testing (--prompt "...") or continuous interactive chat loop.
- Supports lightweight mock testing (--mock) for local testing without downloading GPU weights.

Usage:
    # 1. Interactive chat with baseline model (in Colab or with GPU):
    python ml/scripts/chat_interactive.py

    # 2. Chat with Stage 7A candidate (3B):
    python ml/scripts/chat_interactive.py --variant stage7a_qwen2.5_3b

    # 3. Chat with full 4-layer cognitive pipeline (Context + Memory + Orchestrator):
    python ml/scripts/chat_interactive.py --pipeline

    # 4. Quick single-turn test:
    python ml/scripts/chat_interactive.py --prompt "konsa laptop hai tere pass?"

    # 5. Local testing on laptop without GPU weights (simulated mock):
    python ml/scripts/chat_interactive.py --mock
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root and ml root in sys.path
ml_root = Path(__file__).resolve().parents[1]
repo_root = ml_root.parent
for p in [str(repo_root), str(ml_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.src.context.context_analyzer import ContextAnalyzer
from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator
from ml.src.memory.embedding import DeterministicMockEmbedding
from ml.src.memory.memory_ranker import MemoryRanker
from ml.src.memory.memory_retriever import MemoryRetriever
from ml.src.memory.memory_schema import ImportanceLevel, Memory, MemoryType
from ml.src.memory.memory_store import InMemoryMemoryStore
from ml.src.orchestrator.orchestrator import ConversationOrchestrator
from ml.src.orchestrator.orchestrator_schema import ConversationRequest

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def build_default_memory_store() -> InMemoryMemoryStore:
    """Pre-seeds standard core persona memories for Vivek."""
    embedding_model = DeterministicMockEmbedding(dimension=64)
    store = InMemoryMemoryStore(embedding_model=embedding_model)

    core_memories = [
        Memory(
            id="mem_laptop_nitro",
            persona_id="vivek",
            memory_type=MemoryType.FACT,
            content="Owns an Acer Nitro laptop",
            topic="technology",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem_college_noida",
            persona_id="vivek",
            memory_type=MemoryType.FACT,
            content="Studies B.Tech CSE in college located in Noida",
            topic="college",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem_gaming_bgmi",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Plays BGMI regularly with friends late at night",
            topic="gaming",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem_food_canteen",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Frequently hangs out with friends at the college canteen",
            topic="food",
            importance=ImportanceLevel.MEDIUM,
        ),
        Memory(
            id="mem_coding_python",
            persona_id="vivek",
            memory_type=MemoryType.FACT,
            content="Focuses on backend engineering and Python machine learning",
            topic="technology",
            importance=ImportanceLevel.MEDIUM,
        ),
    ]

    for m in core_memories:
        store.save(m)

    return store


def generate_pipeline_turn(
    messages: List[Dict[str, str]],
    generator: PersonaGenerator,
    context_analyzer: ContextAnalyzer,
    retriever: MemoryRetriever,
    orchestrator: ConversationOrchestrator,
    gen_cfg: GenerationConfig,
    show_internal_reasoning: bool = True,
) -> str:
    """Routes turn through full 4-layer cognitive architecture."""
    # 1. Context Engine
    t0 = time.perf_counter()
    context = context_analyzer.analyze(messages)
    ctx_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Memory Engine
    t1 = time.perf_counter()
    retrieved = retriever.retrieve(
        persona_id="vivek",
        conversation_context=messages,
        context_analysis=context,
        top_k=2,
    )
    mem_ms = (time.perf_counter() - t1) * 1000.0
    memory_snippets = [m.memory.content for m in retrieved]

    # 3. Conversation Orchestrator
    t2 = time.perf_counter()
    plan = orchestrator.plan(ConversationRequest(
        persona_id="vivek",
        conversation_id="interactive_chat",
        messages=messages,
        context=context,
        memories=retrieved,
    ))
    orch_ms = (time.perf_counter() - t2) * 1000.0

    if show_internal_reasoning:
        print(f"\n  [Cognitive Inspection]")
        print(f"  • Context Engine : Topic='{context.topic.value}', Intent='{context.user_intent.value}', Ambiguity='{context.ambiguity.value}' ({ctx_ms:.1f}ms)")
        if memory_snippets:
            print(f"  • Memory Engine  : Retrieved {memory_snippets} ({mem_ms:.1f}ms)")
        print(f"  • Orchestrator   : Strategy='{plan.response_strategy.value}', Tone='{plan.tone.value}' ({orch_ms:.1f}ms)")
        print(f"  • Instruction    : \"{plan.generation_instruction}\"")

    # 4. Generator
    context_messages = list(messages)
    context_messages.insert(
        0,
        {"role": "system", "content": f"Tone: {plan.tone.value}. Strategy: {plan.response_strategy.value}. {plan.generation_instruction}"}
    )

    t3 = time.perf_counter()
    reply = generator.generate(context_messages, generation_config=gen_cfg)
    gen_ms = (time.perf_counter() - t3) * 1000.0

    if show_internal_reasoning:
        total_ms = ctx_ms + mem_ms + orch_ms + gen_ms
        print(f"  • Generation     : {len(reply.split())} words ({gen_ms:.1f}ms) | Total E2E: {total_ms:.1f}ms\n")

    return reply


def main():
    parser = argparse.ArgumentParser(description="Interactive Chat with Persona Engine")
    parser.add_argument(
        "--variant",
        type=str,
        default="stage5_v1",
        choices=["stage5_v1", "stage7a_qwen2.5_3b", "base_qwen2.5_1.5b"],
        help="Model variant to test (default: stage5_v1, or base_qwen2.5_1.5b for base model without adapter)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Enable full 4-layer cognitive pipeline (Context + Memory + Orchestrator + Generator)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use simulated mock handle (for lightweight testing without downloading weights)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test Vivek's response directly",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max output tokens (default: 64)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PERSONA ENGINE — INTERACTIVE CHAT INSPECTOR")
    print("=" * 70)
    print(f"Target Variant : {args.variant}")
    print(f"Cognitive Mode : {'Full 4-Layer Pipeline' if args.pipeline else 'Direct Persona Model'}")
    print(f"Mock Mode      : {args.mock}")
    print(f"Temperature    : {args.temperature}")
    print(f"Max New Tokens : {args.max_tokens}")
    print("=" * 70)

    # Initialize Generator
    generator = PersonaGenerator(model_id=args.variant)

    print("\nLoading model and LoRA adapter...")
    try:
        generator.load_model(mock=args.mock)
    except FileNotFoundError as fnf_err:
        print("\n" + "!" * 70)
        print("ADAPTER WEIGHTS REQUIRED")
        print(str(fnf_err))
        print("\nNote: Real weights must be loaded in Google Colab GPU runtime or mounted via")
        print("ADAPTER_OVERRIDE_PATH. To run locally without weights, use: --mock")
        print("!" * 70 + "\n")
        sys.exit(1)

    info = ModelLoader.get_info()
    print(f"  Model Loaded   : {info['model_id']}")
    print(f"  Base Model     : {info['base_model']}")
    print(f"  Adapter Path   : {info['adapter_path']}")
    print(f"  Adapter Applied: {info['adapter_loaded']}")
    print(f"  Device         : {info['device']}")
    print(f"  Quantization   : {info['quantization']}")
    print("=" * 70)

    # Setup pipeline components if requested
    gen_cfg = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
    ctx_analyzer = ContextAnalyzer() if args.pipeline else None
    mem_store = build_default_memory_store() if args.pipeline else None
    embedding_model = DeterministicMockEmbedding(dimension=64) if args.pipeline else None
    mem_retriever = MemoryRetriever(store=mem_store, ranker=MemoryRanker(), embedding_model=embedding_model) if args.pipeline else None
    orchestrator = ConversationOrchestrator(context_analyzer=ctx_analyzer) if args.pipeline else None

    # Single-prompt test
    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        print(f"\nYou: {args.prompt}")
        if args.pipeline:
            reply = generate_pipeline_turn(
                messages, generator, ctx_analyzer, mem_retriever, orchestrator, gen_cfg, show_internal_reasoning=True
            )
            print(f"Vivek: {reply}\n")
        else:
            t0 = time.perf_counter()
            reply = generator.generate(messages, generation_config=gen_cfg)
            dur_ms = (time.perf_counter() - t0) * 1000.0
            print(f"Vivek: {reply} ({dur_ms:.1f}ms)\n")
        return

    # Check if stdin is an interactive TTY
    if not sys.stdin.isatty():
        print("\n[NOTE] Non-interactive environment detected. Running sample test dialogues:")
        sample_prompts = [
            "bhai kya scene hai aaj ka?",
            "konsa laptop hai tere pass?",
            "canteen me baitha hu kab aayega?",
            "khelega kya aaj raat bgmi?",
        ]
        for p in sample_prompts:
            print(f"\nYou: {p}")
            messages = [{"role": "user", "content": p}]
            if args.pipeline:
                reply = generate_pipeline_turn(
                    messages, generator, ctx_analyzer, mem_retriever, orchestrator, gen_cfg, show_internal_reasoning=True
                )
                print(f"Vivek: {reply}\n")
            else:
                reply = generator.generate(messages, generation_config=gen_cfg)
                print(f"Vivek: {reply}")
        return

    # Interactive Chat Loop
    print("\nChat with Vivek! (Type 'exit', 'quit', or 'reset' to manage session)\n")
    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("\nExiting chat session.")
                break
            if user_input.lower() in {"reset", "clear"}:
                history = []
                print("[Conversation history cleared]\n")
                continue

            history.append({"role": "user", "content": user_input})

            if args.pipeline:
                reply = generate_pipeline_turn(
                    history, generator, ctx_analyzer, mem_retriever, orchestrator, gen_cfg, show_internal_reasoning=True
                )
                print(f"Vivek: {reply}\n")
            else:
                t0 = time.perf_counter()
                reply = generator.generate(history, generation_config=gen_cfg)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                print(f"Vivek: {reply} ({lat_ms:.1f}ms)\n")

            history.append({"role": "assistant", "content": reply})

            # Keep sliding window of last 10 turns
            if len(history) > 10:
                history = history[-10:]

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


if __name__ == "__main__":
    main()
