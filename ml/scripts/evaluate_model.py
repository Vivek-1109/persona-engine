#!/usr/bin/env python3
"""
Persona Engine — Model Evaluation Script
Evaluates model responses against persona ground truth dialogues.
Calculates length similarity, vocabulary overlap, emoji consistency, and punctuation alignment.

Usage:
    python ml/scripts/evaluate_model.py --eval-data ml/data/sample/sample_conversations.jsonl
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.data.loader import load_jsonl
from src.evaluation.evaluator import PersonaEvaluator
from src.inference.generator import PersonaGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate persona model responses against reference conversational style."
    )
    parser.add_argument(
        "--eval-data",
        "-d",
        type=str,
        default=str(ml_root / "data" / "sample" / "sample_conversations.jsonl"),
        help="Path to evaluation dataset file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to save output evaluation JSON report.",
    )
    parser.add_argument(
        "--max-samples",
        "-m",
        type=int,
        default=10,
        help="Maximum samples to evaluate.",
    )

    args = parser.parse_args()
    data_path = Path(args.eval_data)

    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}", file=sys.stderr)
        sys.exit(1)

    records = load_jsonl(data_path)
    generator = PersonaGenerator()
    evaluator = PersonaEvaluator()

    pairs = []
    for conv in records[: args.max_samples]:
        messages = conv.get("messages", [])
        for i, msg in enumerate(messages):
            if msg.get("speaker") == "persona" and i > 0:
                context = messages[:i]
                prompt = messages[i - 1].get("text", "")
                reference = msg.get("text", "")
                # Generate model response
                generated = generator.generate(context)
                pairs.append({
                    "prompt": prompt,
                    "reference": reference,
                    "generated": generated,
                })

    print(f"Evaluating {len(pairs)} turn samples against reference persona style...\n")
    report = evaluator.evaluate(pairs)
    print(report.summary_text())

    if args.output:
        out_path = Path(args.output)
        evaluator.save_report(report, out_path)
        print(f"\nSaved evaluation metrics report to: {out_path}")


if __name__ == "__main__":
    main()
