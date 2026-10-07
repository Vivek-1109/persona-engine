#!/usr/bin/env python3
"""
Persona Engine — Dataset Validation Script
Validates a JSONL conversation file against schemas and business rules.

Usage:
    python ml/scripts/validate_dataset.py --input ml/data/sample/sample_conversations.jsonl
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

from src.data.validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(
        description="Validate Persona Engine conversational dataset files."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(ml_root / "data" / "sample" / "sample_conversations.jsonl"),
        help="Path to JSONL dataset file to validate.",
    )
    parser.add_argument(
        "--schema",
        "-s",
        type=str,
        default=None,
        help="Optional path to custom JSON schema file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed list of warnings and errors.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    print(f"Validating dataset: {input_path}")
    validator = DatasetValidator(schema_path=args.schema)
    result = validator.validate_file(input_path)

    print("\n" + result.summary())

    if args.verbose or not result.is_valid:
        if result.issues:
            print("\nIssues found:")
            for issue in result.issues:
                prefix = f"[{issue.severity}]"
                conv_str = f" Conv: {issue.conversation_id}" if issue.conversation_id else ""
                msg_idx_str = f" Msg: {issue.message_index}" if issue.message_index is not None else ""
                print(f"  {prefix}{conv_str}{msg_idx_str}: {issue.message}")

    if not result.is_valid:
        sys.exit(1)
    else:
        print("\nAll validation checks passed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
