#!/usr/bin/env python3
"""
Persona Engine — Persona Dataset Validator
Validates train.jsonl, val.jsonl, and test.jsonl against strict criteria:
- Valid JSONL format
- Valid message roles (system, user, assistant)
- Assistant is always the final target turn
- No empty target responses
- No malformed examples
- No train/val/test conversation overlap (Zero data leakage)
- No duplicate example IDs
- No impossible/malformed timestamps
- No missing conversation IDs
- No accidental metadata leakage inside message content
- Expected speaker mapping

Exits with non-zero status code if any critical validation check fails.
"""

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml root in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))


class PersonaDatasetValidator:
    """Strict validator for Persona Engine fine-tuning datasets."""

    ALLOWED_ROLES = {"system", "user", "assistant"}

    def __init__(
        self,
        training_dir: Path,
        expected_speaker: str = "Vivek",
        train_file_name: str = "train.jsonl",
    ):
        self.training_dir = training_dir
        self.expected_speaker = expected_speaker
        self.train_file_name = train_file_name
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and parse JSONL records, catching line-level syntax errors."""
        records: List[Dict[str, Any]] = []
        if not file_path.exists():
            self.errors.append(f"File missing: {file_path}")
            return records

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                clean = line.strip()
                if not clean:
                    continue
                try:
                    data = json.loads(clean)
                    records.append(data)
                except json.JSONDecodeError as e:
                    self.errors.append(f"{file_path.name}:{line_no} Invalid JSON: {e}")

        return records

    def validate_example(
        self, example: Dict[str, Any], file_name: str, index: int
    ) -> bool:
        """Validate an individual example for schema and content integrity."""
        valid = True
        prefix = f"[{file_name} Ex#{index}]"

        # Check top-level keys
        if "messages" not in example:
            self.errors.append(f"{prefix} Missing 'messages' field")
            return False

        messages = example["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            self.errors.append(f"{prefix} 'messages' must be a list with at least 2 entries")
            return False

        # Validate message turns
        for m_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                self.errors.append(f"{prefix} Message #{m_idx} is not a dict")
                valid = False
                continue

            role = msg.get("role")
            content = msg.get("content")

            if role not in self.ALLOWED_ROLES:
                self.errors.append(f"{prefix} Message #{m_idx} invalid role: {role}")
                valid = False

            if not isinstance(content, str) or not content.strip():
                self.errors.append(f"{prefix} Message #{m_idx} ({role}) empty or non-string content")
                valid = False

            # Check for accidental metadata leakage into content
            if content and any(marker in content for marker in ['"metadata":', '"conversation_id":', '"example_id":']):
                self.errors.append(f"{prefix} Potential metadata leakage detected in message #{m_idx}")
                valid = False

            # Check for WhatsApp protocol artifact "Waiting for this message"
            if content and "waiting for this message" in content.lower():
                self.errors.append(f"{prefix} Protocol artifact detected in message #{m_idx}: '{content}'")
                valid = False

        # Final message must ALWAYS be assistant
        if messages[-1].get("role") != "assistant":
            self.errors.append(f"{prefix} Final message role must be 'assistant', got: {messages[-1].get('role')}")
            valid = False

        # Metadata validation
        metadata = example.get("metadata")
        if metadata is not None and isinstance(metadata, dict):
            conv_id = metadata.get("conversation_id")
            if not conv_id:
                self.errors.append(f"{prefix} Missing 'conversation_id' in metadata")
                valid = False

            speaker = metadata.get("speaker")
            if speaker and speaker != self.expected_speaker:
                self.warnings.append(f"{prefix} Speaker '{speaker}' != expected '{self.expected_speaker}'")

            ts = metadata.get("timestamp")
            if ts:
                try:
                    datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    self.warnings.append(f"{prefix} Non-standard timestamp format: {ts}")

        return valid

    def validate_split(
        self, split_name: str, records: List[Dict[str, Any]]
    ) -> Tuple[Set[str], Set[str]]:
        """Validate a full split and return sets of (example_ids, conversation_ids)."""
        seen_example_ids: Set[str] = set()
        conv_ids: Set[str] = set()

        for idx, ex in enumerate(records):
            self.validate_example(ex, split_name, idx)

            meta = ex.get("metadata", {})
            ex_id = meta.get("example_id")
            if ex_id:
                if ex_id in seen_example_ids:
                    self.errors.append(f"[{split_name}] Duplicate example_id: {ex_id}")
                seen_example_ids.add(ex_id)

            conv_id = meta.get("conversation_id")
            if conv_id:
                conv_ids.add(conv_id)

        return seen_example_ids, conv_ids

    def run_all(self) -> bool:
        """Execute full dataset validation suite."""
        print(f"[VALIDATION] Scanning training directory: {self.training_dir}")
        train_path = self.training_dir / self.train_file_name
        val_path = self.training_dir / "val.jsonl"
        test_path = self.training_dir / "test.jsonl"

        train_records = self.load_jsonl(train_path)
        val_records = self.load_jsonl(val_path)
        test_records = self.load_jsonl(test_path)

        print(f"  Loaded Train ({self.train_file_name}): {len(train_records):,} examples")
        print(f"  Loaded Validation: {len(val_records):,} examples")
        print(f"  Loaded Test:       {len(test_records):,} examples")

        if not train_records or not val_records or not test_records:
            self.errors.append("One or more required dataset files (train/val/test) are empty or missing.")

        # Validate each split
        train_ex_ids, train_convs = self.validate_split("train.jsonl", train_records)
        val_ex_ids, val_convs = self.validate_split("val.jsonl", val_records)
        test_ex_ids, test_convs = self.validate_split("test.jsonl", test_records)

        # Cross-split example ID uniqueness
        dup_train_val_ex = train_ex_ids & val_ex_ids
        dup_train_test_ex = train_ex_ids & test_ex_ids
        dup_val_test_ex = val_ex_ids & test_ex_ids

        if dup_train_val_ex:
            self.errors.append(f"Cross-split duplicate example IDs between train and val: {len(dup_train_val_ex)}")
        if dup_train_test_ex:
            self.errors.append(f"Cross-split duplicate example IDs between train and test: {len(dup_train_test_ex)}")
        if dup_val_test_ex:
            self.errors.append(f"Cross-split duplicate example IDs between val and test: {len(dup_val_test_ex)}")

        # Zero Data Leakage Check (Conversation ID disjointness)
        leak_train_val = train_convs & val_convs
        leak_train_test = train_convs & test_convs
        leak_val_test = val_convs & test_convs

        if leak_train_val:
            self.errors.append(f"LEAKAGE DETECTED: {len(leak_train_val)} conversations shared between train and validation!")
        if leak_train_test:
            self.errors.append(f"LEAKAGE DETECTED: {len(leak_train_test)} conversations shared between train and test!")
        if leak_val_test:
            self.errors.append(f"LEAKAGE DETECTED: {len(leak_val_test)} conversations shared between validation and test!")

        # Print report
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)
        print(f"Train conversations:      {len(train_convs):,}")
        print(f"Validation conversations: {len(val_convs):,}")
        print(f"Test conversations:       {len(test_convs):,}")
        print(f"Total Unique Convs:       {len(train_convs | val_convs | test_convs):,}")
        print("-" * 60)
        print(f"Train ∩ Val Overlap:      {len(leak_train_val)} convs")
        print(f"Train ∩ Test Overlap:     {len(leak_train_test)} convs")
        print(f"Val ∩ Test Overlap:       {len(leak_val_test)} convs")
        print("-" * 60)

        if self.warnings:
            print(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                print(f"  [WARN] {w}")

        if not self.errors:
            print("STATUS: ALL VALIDATION CHECKS PASSED PERFECTLY (0 ERRORS)")
            print("=" * 60)
            return True
        else:
            print(f"STATUS: VALIDATION FAILED WITH {len(self.errors)} ERROR(S)")
            for err in self.errors[:10]:
                print(f"  [ERROR] {err}")
            print("=" * 60)
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Persona Engine datasets.")
    parser.add_argument(
        "--train-file",
        type=str,
        default="train.jsonl",
        help="Name of the training file to validate (e.g., train.jsonl or stage4_train.jsonl)",
    )
    args = parser.parse_args()

    training_dir = ml_root / "data" / "training"
    validator = PersonaDatasetValidator(
        training_dir=training_dir,
        expected_speaker="Vivek",
        train_file_name=args.train_file,
    )
    passed = validator.run_all()
    if not passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
