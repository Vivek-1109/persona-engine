#!/usr/bin/env python3
"""
Persona Engine — Stage 5: Dataset Audit & Integrity Validator
Performs comprehensive integrity validation and statistical profiling on Stage 5 datasets:
1. Zero missing target responses
2. Zero target-response modifications (exact 1-to-1 match with source Stage 4 datasets)
3. Zero train/val/test conversation overlap (strictly disjoint)
4. Zero duplicate dialogue examples
5. Complete statistical profiling:
   - Behavioral distributions (language, tone, response_type, has_slang, has_emoji, is_question)
   - Topic distributions (9 categories)
   - Context-depth distributions (1, 2, 4, 6 turns)
   - Emoji distributions (rates, frequencies, top emojis)
   - Response-length distributions (chars, words, category breakdown)
   - Response-type distributions
6. Generates ml/data/stage5/reports/stage5_dataset_audit.md
"""

from collections import Counter, defaultdict
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata

# Set UTF-8 encoding for Windows console if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AuditStage5Dataset")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                records.append(json.loads(clean))
    return records


def extract_emojis(text: str) -> List[str]:
    emojis = []
    for char in text:
        cp = ord(char)
        if (
            unicodedata.category(char) in ("So", "Sk")
            or 0x1F300 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or 0x1F600 <= cp <= 0x1F64F
            or 0x1F680 <= cp <= 0x1F6FF
            or 0x2B50 <= cp <= 0x2B55
        ):
            emojis.append(char)
    return emojis


def audit_stage5_dataset():
    print("=" * 70)
    print("STAGE 5 — DATASET AUDIT & INTEGRITY VALIDATION")
    print("=" * 70)

    stage5_dir = ml_root / "data/stage5"
    stage4_dir = ml_root / "data/training"
    reports_dir = stage5_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Paths
    s5_train_path = stage5_dir / "training/stage5_train.jsonl"
    s5_val_path = stage5_dir / "training/val.jsonl"
    s5_test_path = stage5_dir / "training/test.jsonl"

    src_train_path = stage4_dir / "stage4_train.jsonl"
    src_val_path = stage4_dir / "val.jsonl"
    src_test_path = stage4_dir / "test.jsonl"

    ambiguous_path = stage5_dir / "benchmarks/ambiguous_prompts.jsonl"

    assert s5_train_path.is_file(), f"Missing {s5_train_path}"
    assert s5_val_path.is_file(), f"Missing {s5_val_path}"
    assert s5_test_path.is_file(), f"Missing {s5_test_path}"

    # 2. Load Datasets
    print("\n[1/5] Loading Stage 5 datasets and Stage 4 reference datasets...")
    s5_train = load_jsonl(s5_train_path)
    s5_val = load_jsonl(s5_val_path)
    s5_test = load_jsonl(s5_test_path)

    src_train = load_jsonl(src_train_path)
    src_val = load_jsonl(src_val_path)
    src_test = load_jsonl(src_test_path)

    ambiguous_records = load_jsonl(ambiguous_path) if ambiguous_path.is_file() else []

    print(f"  Stage 5 Train : {len(s5_train):,} (Source: {len(src_train):,})")
    print(f"  Stage 5 Val   : {len(s5_val):,} (Source: {len(src_val):,})")
    print(f"  Stage 5 Test  : {len(s5_test):,} (Source: {len(src_test):,})")

    # 3. Integrity Assertions
    print("\n[2/5] Running strict integrity checks...")
    errors: List[str] = []

    splits = [("train", s5_train, src_train), ("val", s5_val, src_val), ("test", s5_test, src_test)]

    # A. Check 1-to-1 match with source
    for name, s5_data, src_data in splits:
        if len(s5_data) != len(src_data):
            errors.append(f"Split {name} length mismatch: Stage 5 has {len(s5_data)}, source has {len(src_data)}")

        for idx, (s5_r, src_r) in enumerate(zip(s5_data, src_data)):
            src_tgt = src_r["messages"][-1]["content"]
            s5_tgt = s5_r.get("target_response", "")

            # Missing target response
            if not s5_tgt or not s5_tgt.strip():
                errors.append(f"Missing target response at {name}[{idx}]")

            # Target response modification check
            if s5_tgt != src_tgt:
                errors.append(f"Target response modified at {name}[{idx}]: '{s5_tgt}' != '{src_tgt}'")

            # Metadata preservation
            if s5_r.get("conversation_id") != src_r["metadata"].get("conversation_id"):
                errors.append(f"conversation_id mismatch at {name}[{idx}]")
            if s5_r.get("target_message_id") != src_r["metadata"].get("target_message_id"):
                errors.append(f"target_message_id mismatch at {name}[{idx}]")

            # Check metadata fields exist
            for req_field in ["behavioral_metadata", "topic_metadata", "context_metadata", "context"]:
                if req_field not in s5_r:
                    errors.append(f"Missing required field '{req_field}' at {name}[{idx}]")

    # B. Conversation overlap check
    train_convs = {r["conversation_id"] for r in s5_train}
    val_convs = {r["conversation_id"] for r in s5_val}
    test_convs = {r["conversation_id"] for r in s5_test}

    train_val_overlap = train_convs & val_convs
    train_test_overlap = train_convs & test_convs
    val_test_overlap = val_convs & test_convs

    if train_val_overlap:
        errors.append(f"Conversation overlap between train and val: {len(train_val_overlap)} convs")
    if train_test_overlap:
        errors.append(f"Conversation overlap between train and test: {len(train_test_overlap)} convs")
    if val_test_overlap:
        errors.append(f"Conversation overlap between val and test: {len(val_test_overlap)} convs")

    # C. Duplicate example and intra-conversation check
    surface_repeats: Dict[str, int] = {}
    for name, s5_data, _ in splits:
        seen_example_ids: Set[str] = set()
        seen_conv_keys: Set[Tuple[str, Any, int]] = set()
        seen_surfaces: Counter = Counter()

        dup_ex_count = 0
        dup_conv_count = 0

        for idx, r in enumerate(s5_data):
            ex_id = r.get("metadata", {}).get("example_id") or f"{r.get('conversation_id')}_{r.get('target_message_id')}_{len(r.get('context', []))}"
            if ex_id in seen_example_ids:
                dup_ex_count += 1
            seen_example_ids.add(ex_id)

            conv_key = (r.get("conversation_id"), r.get("target_message_id"), len(r.get("context", [])))
            if conv_key in seen_conv_keys:
                dup_conv_count += 1
            seen_conv_keys.add(conv_key)

            ctx_str = " | ".join(f"{m.get('role')}:{m.get('content')}" for m in r.get("context", []))
            surf_key = f"{ctx_str} --> {r.get('target_response')}"
            seen_surfaces[surf_key] += 1

        if dup_ex_count > 0:
            errors.append(f"Found {dup_ex_count} duplicate example IDs in {name} split")
        if dup_conv_count > 0:
            errors.append(f"Found {dup_conv_count} duplicate dialogue records within same conversation in {name} split")

        surface_repeats[name] = sum(count - 1 for count in seen_surfaces.values() if count > 1)

    if errors:
        print(f"\n[FAIL] Found {len(errors)} integrity errors:")
        for e in errors[:10]:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("  [PASS] Zero missing target responses.")
        print("  [PASS] Zero target-response modifications (100% exact match).")
        print("  [PASS] Zero train/val/test conversation leakage.")
        print("  [PASS] Zero duplicate dialogue example records.")
        print("  [PASS] Zero intra-conversation duplicate examples.")
        print(f"  [INFO] Natural cross-session dialogue collisions: train={surface_repeats['train']}, val={surface_repeats['val']}, test={surface_repeats['test']}")

    # 4. Statistical Distributions & Profiling
    print("\n[3/5] Calculating distributions across Stage 5 datasets...")
    all_s5 = s5_train + s5_val + s5_test

    # Behavioral
    languages = Counter(r["behavioral_metadata"]["language"] for r in all_s5)
    tones = Counter(r["behavioral_metadata"]["tone"] for r in all_s5)
    response_types = Counter(r["behavioral_metadata"]["response_type"] for r in all_s5)
    length_cats = Counter(r["behavioral_metadata"]["response_length"]["category"] for r in all_s5)
    has_slang_count = sum(1 for r in all_s5 if r["behavioral_metadata"]["has_slang"])
    has_emoji_count = sum(1 for r in all_s5 if r["behavioral_metadata"]["has_emoji"])
    is_question_count = sum(1 for r in all_s5 if r["behavioral_metadata"]["is_question"])

    # Topic
    topics = Counter(r["topic_metadata"]["primary_topic"] for r in all_s5)

    # Context Depth
    context_depths = Counter(r["context_metadata"]["context_depth"] for r in all_s5)
    states = Counter(r["context_metadata"]["conversation_state"] for r in all_s5)

    # Emoji Profiling
    all_emojis: List[str] = []
    for r in all_s5:
        all_emojis.extend(extract_emojis(r["target_response"]))
    emoji_counts = Counter(all_emojis)

    # Response Length Profiling
    char_lengths = [r["behavioral_metadata"]["response_length"]["char_count"] for r in all_s5]
    word_lengths = [r["behavioral_metadata"]["response_length"]["word_count"] for r in all_s5]

    avg_chars = round(sum(char_lengths) / len(char_lengths), 2)
    avg_words = round(sum(word_lengths) / len(word_lengths), 2)
    median_chars = sorted(char_lengths)[len(char_lengths) // 2]
    median_words = sorted(word_lengths)[len(word_lengths) // 2]

    # 5. Generate Audit Report Markdown
    print("\n[4/5] Generating Stage 5 Dataset Audit Report...")
    report_file = reports_dir / "stage5_dataset_audit.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Stage 5 — Dataset Audit & Integrity Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Total Stage 5 Examples:** {len(all_s5):,}  \n")
        f.write(f"**Split Counts:** Train: {len(s5_train):,} | Val: {len(s5_val):,} | Test: {len(s5_test):,}  \n")
        f.write(f"**Unique Conversations:** Train: {len(train_convs)} | Val: {len(val_convs)} | Test: {len(test_convs)} (Total: {len(train_convs | val_convs | test_convs)})  \n\n")

        f.write("## 1. Integrity Verification Summary\n\n")
        f.write("| Integrity Check | Expected | Actual | Status |\n")
        f.write("|:---|:---:|:---:|:---:|\n")
        f.write(f"| Missing Target Responses | 0 | 0 | **PASS** |\n")
        f.write(f"| Target Response Modifications | 0 | 0 | **PASS** |\n")
        f.write(f"| Train ∩ Val Conversation Overlap | 0 | 0 | **PASS** |\n")
        f.write(f"| Train ∩ Test Conversation Overlap | 0 | 0 | **PASS** |\n")
        f.write(f"| Val ∩ Test Conversation Overlap | 0 | 0 | **PASS** |\n")
        f.write(f"| Duplicate Dialogue Examples | 0 | 0 | **PASS** |\n")
        f.write(f"| Field Completeness (Behavioral, Topic, Context) | 100% | 100% | **PASS** |\n\n")

        f.write("## 2. Behavioral Distributions\n\n")
        f.write("### Language Distribution\n")
        f.write("| Language | Count | Percentage |\n")
        f.write("|:---|:---:|:---:|\n")
        for lang, count in languages.most_common():
            f.write(f"| `{lang}` | {count:,} | {count / len(all_s5) * 100:.2f}% |\n")

        f.write("\n### Tone Distribution\n")
        f.write("| Tone | Count | Percentage |\n")
        f.write("|:---|:---:|:---:|\n")
        for tone, count in tones.most_common():
            f.write(f"| `{tone}` | {count:,} | {count / len(all_s5) * 100:.2f}% |\n")

        f.write("\n### Response Type Distribution\n")
        f.write("| Response Type | Count | Percentage |\n")
        f.write("|:---|:---:|:---:|\n")
        for rtype, count in response_types.most_common():
            f.write(f"| `{rtype}` | {count:,} | {count / len(all_s5) * 100:.2f}% |\n")

        f.write("\n### Stylistic Markers\n")
        f.write("| Marker | Count | Percentage |\n")
        f.write("|:---|:---:|:---:|\n")
        f.write(f"| Has Slang / Colloquialisms | {has_slang_count:,} | {has_slang_count / len(all_s5) * 100:.2f}% |\n")
        f.write(f"| Has Emoji | {has_emoji_count:,} | {has_emoji_count / len(all_s5) * 100:.2f}% |\n")
        f.write(f"| Is Question | {is_question_count:,} | {is_question_count / len(all_s5) * 100:.2f}% |\n\n")

        f.write("## 3. Topic Distributions\n\n")
        f.write("| Topic Category | Count | Percentage |\n")
        f.write("|:---|:---:|:---:|\n")
        for topic, count in topics.most_common():
            f.write(f"| `{topic}` | {count:,} | {count / len(all_s5) * 100:.2f}% |\n")

        f.write("\n## 4. Context & Structural Dynamics\n\n")
        f.write("### Context Depth Distribution\n")
        f.write("| Context Messages | Count | Percentage |\n")
        f.write("|:---:|:---:|:---:|\n")
        for depth, count in sorted(context_depths.items()):
            f.write(f"| {depth} | {count:,} | {count / len(all_s5) * 100:.2f}% |\n")

        f.write("\n### Inferred Conversation State\n")
        f.write("| State | Count | Percentage |\n")
        f.write("|:---|:---:|:---:|\n")
        for state, count in states.most_common():
            f.write(f"| `{state}` | {count:,} | {count / len(all_s5) * 100:.2f}% |\n")

        f.write("\n## 5. Emoji & Length Statistics\n\n")
        f.write(f"- **Total Emojis in Dataset:** {len(all_emojis):,}\n")
        f.write(f"- **Emoji Occurrence Rate:** {has_emoji_count / len(all_s5) * 100:.2f}% of messages contain emojis\n")
        f.write(f"- **Top Emojis:** {', '.join(f'{e} ({c})' for e, c in emoji_counts.most_common(10))}\n\n")

        f.write("### Response Length Metrics\n")
        f.write(f"- **Average Characters / Message:** {avg_chars} (Median: {median_chars})\n")
        f.write(f"- **Average Words / Message:** {avg_words} (Median: {median_words})\n\n")
        f.write("| Length Category | Count | Percentage |\n")
        f.write("|:---|:---:|:---:|\n")
        for cat, count in length_cats.most_common():
            f.write(f"| `{cat}` | {count:,} | {count / len(all_s5) * 100:.2f}% |\n")

        f.write("\n## 6. Ambiguous Prompts Benchmark Summary\n\n")
        f.write(f"- **Total Ambiguous Prompts Identified:** {len(ambiguous_records):,}\n")
        f.write("| Prompt | Distinct Target Responses | Total Occurrences |\n")
        f.write("|:---|:---:|:---:|\n")
        for amb in ambiguous_records[:10]:
            f.write(f"| `{amb['prompt']}` | {amb['variant_count']} | {amb['total_occurrences']} |\n")

    print(f"  Audit report successfully generated at: {report_file}")

    # 6. Console Report
    print("\n" + "=" * 60)
    print("STAGE 5 DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total Stage 5 Examples : {len(all_s5):,}")
    print(f"Train / Val / Test     : {len(s5_train):,} / {len(s5_val):,} / {len(s5_test):,}")
    print(f"Integrity Status       : ALL CHECKS PASSED (0 ERRORS)")
    print("-" * 60)
    print("Top Languages          :", dict(languages.most_common(3)))
    print("Top Tones              :", dict(tones.most_common(4)))
    print("Top Topics             :", dict(topics.most_common(5)))
    print("Top Response Types     :", dict(response_types.most_common(4)))
    print(f"Emoji Presence Rate    : {has_emoji_count / len(all_s5) * 100:.2f}%")
    print(f"Avg Response Length    : {avg_chars} chars ({avg_words} words)")
    print(f"Ambiguous Prompts      : {len(ambiguous_records)} prompts identified")
    print("=" * 60)


if __name__ == "__main__":
    audit_stage5_dataset()
