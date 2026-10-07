#!/usr/bin/env python3
"""
Persona Engine — Dataset Analysis Script
Analyzes conversation characteristics, lengths, speaker turn dynamics,
emojis, Hinglish patterns, and vocabulary statistics.

Usage:
    python ml/scripts/analyze_dataset.py --input ml/data/sample/sample_conversations.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml root is in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.analysis.dataset_analyzer import analyze_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Analyze conversational dataset metrics and style attributes."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(ml_root / "data" / "sample" / "sample_conversations.jsonl"),
        help="Path to JSONL dataset file to analyze.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save JSON analysis report.",
    )
    parser.add_argument(
        "--speaker",
        "-s",
        type=str,
        default="persona",
        help="Target persona speaker identifier (default: persona).",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Dataset file not found at {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing dataset: {input_path} (Target persona: '{args.speaker}')\n")
    report = analyze_dataset(input_path, target_speaker=args.speaker)

    print(report.summary_text())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis report saved to: {out_path}")


if __name__ == "__main__":
    main()
