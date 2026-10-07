#!/usr/bin/env python3
"""
Persona Engine — Dataset Preprocessing Script
Cleans conversations (preserving personality traits), splits into train/val/test partitions,
and builds instruction-tuned training examples.

Usage:
    python ml/scripts/preprocess_dataset.py --input ml/data/sample/sample_conversations.jsonl --build-training
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

from src.data.cleaner import DataCleaner
from src.data.loader import load_jsonl, save_jsonl
from src.data.splitter import DatasetSplitter
from src.data.validator import DatasetValidator
from src.preprocessing.conversation_builder import ConversationBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess, clean, split, and format Persona Engine datasets."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(ml_root / "data" / "sample" / "sample_conversations.jsonl"),
        help="Path to source conversation JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(ml_root / "data" / "processed"),
        help="Directory to save cleaned conversations.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        default=True,
        help="Split dataset into train, validation, and test subsets.",
    )
    parser.add_argument(
        "--build-training",
        action="store_true",
        default=True,
        help="Generate model-ready chat training examples from the split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Ratio for training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Ratio for validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.10,
        help="Ratio for test split.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    print(f"Loading and validating source dataset: {input_path}")
    validator = DatasetValidator()
    val_res = validator.validate_file(input_path)
    if not val_res.is_valid:
        print(f"Validation failed with {val_res.error_count} errors. Fix errors before preprocessing.")
        sys.exit(1)

    records = load_jsonl(input_path)
    print(f"Loaded {len(records)} conversations.")

    # 1. Clean
    print("Cleaning conversations (personality traits strictly preserved)...")
    cleaner = DataCleaner()
    cleaned = cleaner.clean_dataset(records)
    print(f"Cleaned dataset: {len(cleaned)} usable conversations.")

    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_file = out_dir / f"cleaned_{input_path.name}"
    save_jsonl(cleaned, cleaned_file)
    print(f"Saved cleaned dataset to: {cleaned_file}")

    # 2. Split
    if args.split:
        splitter = DatasetSplitter(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=42,
        )
        splits = splitter.split(cleaned)
        print(f"\n{splits.summary()}")

        # 3. Build training examples if requested
        if args.build_training:
            training_dir = ml_root / "data" / "training"
            eval_dir = ml_root / "data" / "evaluation"
            training_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)

            builder = ConversationBuilder()
            train_examples = builder.build_dataset(splits.train, split="train")
            val_examples = builder.build_dataset(splits.validation, split="validation")
            test_examples = builder.build_dataset(splits.test, split="test")

            train_file = training_dir / "train.jsonl"
            val_file = training_dir / "val.jsonl"
            test_file = eval_dir / "test.jsonl"

            save_jsonl(train_examples, train_file)
            save_jsonl(val_examples, val_file)
            save_jsonl(test_examples, test_file)

            print(f"\nTraining examples generated:")
            print(f"  Train:      {len(train_examples)} examples -> {train_file}")
            print(f"  Validation: {len(val_examples)} examples -> {val_file}")
            print(f"  Test:       {len(test_examples)} examples -> {test_file}")

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()
