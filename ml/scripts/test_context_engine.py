#!/usr/bin/env python3
"""
Integration test script for Stage 6B Context Engine.
Evaluates ContextAnalyzer against the 10 representative ambiguous prompts
established during Stage 5 benchmarking.
"""

from pathlib import Path
import sys

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

# 10 ambiguous conversational examples from Stage 5 evaluation benchmark
AMBIGUOUS_BENCHMARK_EXAMPLES = [
    {
        "id": "1. Aaja (Canteen rendezvous)",
        "messages": [
            {"role": "user", "content": "canteen me milte hai?"},
            {"role": "assistant", "content": "5 min me pohochta hu"},
            {"role": "user", "content": "Aaja"}
        ]
    },
    {
        "id": "2. Khelega? (Post-assignment inquiry)",
        "messages": [
            {"role": "user", "content": "free hai kya abhi?"},
            {"role": "assistant", "content": "assignment submit kar raha tha"},
            {"role": "user", "content": "Khelega?"}
        ]
    },
    {
        "id": "3. Game aaja (Post-dinner gaming)",
        "messages": [
            {"role": "user", "content": "dinner kar liya?"},
            {"role": "assistant", "content": "haa abhi kiya"},
            {"role": "user", "content": "Game aaja"}
        ]
    },
    {
        "id": "4. Thik (Plan confirmation)",
        "messages": [
            {"role": "assistant", "content": "kal 12 baje nikalte hai"},
            {"role": "user", "content": "Thik"}
        ]
    },
    {
        "id": "5. Nhi bhai (College attendance refusal)",
        "messages": [
            {"role": "user", "content": "college aayega?"},
            {"role": "assistant", "content": "kal exam hai"},
            {"role": "user", "content": "Nhi bhai"}
        ]
    },
    {
        "id": "6. Sahi h (Laptop specs reaction)",
        "messages": [
            {"role": "assistant", "content": "acer nitro le liya maine"},
            {"role": "user", "content": "Sahi h"}
        ]
    },
    {
        "id": "7. Bsdk (Playful bot banter)",
        "messages": [
            {"role": "assistant", "content": "terese acha to bot khelta hai 😂"},
            {"role": "user", "content": "Bsdk"}
        ]
    },
    {
        "id": "8. Kya (Story inquiry continuation)",
        "messages": [
            {"role": "assistant", "content": "ek baat sun"},
            {"role": "user", "content": "Kya"}
        ]
    },
    {
        "id": "9. Kaha (Location request)",
        "messages": [
            {"role": "assistant", "content": "bahar nikal jaldi"},
            {"role": "user", "content": "Kaha"}
        ]
    },
    {
        "id": "10. Aaya (Arrival acknowledgement)",
        "messages": [
            {"role": "assistant", "content": "room pe kab tak aayega?"},
            {"role": "user", "content": "Aaya"}
        ]
    }
]


def run_context_engine_integration_test():
    print("=" * 80)
    print("STAGE 6B — CONTEXT ENGINE INTEGRATION BENCHMARK (10 AMBIGUOUS PROMPTS)")
    print("=" * 80)

    for ex in AMBIGUOUS_BENCHMARK_EXAMPLES:
        print(f"\n--- {ex['id']} ---")
        print("Conversation Context:")
        for turn in ex["messages"]:
            print(f"  [{turn['role'].capitalize()}]: {turn['content']}")

        analysis = ContextAnalyzer.analyze(ex["messages"])
        print("\n-> ContextAnalysis Output:")
        print(f"   Topic                   : {analysis.topic.value}")
        print(f"   Conversation State      : {analysis.conversation_state.value}")
        print(f"   User Intent             : {analysis.user_intent.value}")
        print(f"   Ambiguity Level         : {analysis.ambiguity.value}")
        print(f"   Recent Activity         : {analysis.recent_activity}")
        print(f"   Context Depth           : {analysis.context_depth}")
        print(f"   Speaker Alternation Rate: {analysis.speaker_alternation_rate}")

    print("\n" + "=" * 80)
    print("ALL 10 CONTEXT ENGINE INTEGRATION CASES ANALYZED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_context_engine_integration_test()
