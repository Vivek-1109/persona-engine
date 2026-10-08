#!/usr/bin/env python3
"""
Persona Engine — Stage 5: Dataset Engineering Pipeline
Builds the Stage 5 annotated datasets for Context + Behavioral Representation:
- Reads ONLY existing clean Stage 4 training/validation/test datasets.
- Preserves conversation_id, target_message_id, context, and target_response.
- Enriches every record with behavioral_metadata, topic_metadata, and context_metadata.
- Generates ambiguous_prompts.jsonl benchmark file.
- Does NOT train any model.
- Does NOT modify raw data or Stage 4 files.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Set UTF-8 encoding for Windows console if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml root in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.preprocessing.behavioral_features import BehavioralFeatureExtractor
from src.preprocessing.context_features import ContextFeatureExtractor
from src.preprocessing.topic_classifier import DeterministicTopicClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildStage5Dataset")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                records.append(json.loads(clean))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def annotate_record(
    record: Dict[str, Any],
    behavioral_extractor: BehavioralFeatureExtractor,
    topic_classifier: DeterministicTopicClassifier,
    context_extractor: ContextFeatureExtractor,
) -> Dict[str, Any]:
    """
    Transforms a clean Stage 4 record into a rich Stage 5 annotated record.
    Preserves all original fields and adds behavioral, topic, and context metadata.
    """
    messages = record.get("messages", [])
    raw_metadata = record.get("metadata", {})

    conversation_id = raw_metadata.get("conversation_id", "")
    target_message_id = raw_metadata.get("target_message_id", 0)

    # Extract target response and context dialogue
    target_response = messages[-1]["content"] if messages else ""
    context_turns = [m for m in messages[1:-1]]  # Exclude system prompt (index 0) and target (index -1)

    # Preceding user message for context-aware classification
    last_user_text = ""
    for m in reversed(context_turns):
        if m.get("role") == "user":
            last_user_text = m.get("content", "").strip()
            break

    context_combined_text = " ".join([m.get("content", "") for m in context_turns])

    # 1. Behavioral Features
    behav_feat = behavioral_extractor.extract_features(target_response, preceding_user_text=last_user_text)

    # 2. Topic Classification
    topic_feat = topic_classifier.classify(target_response, context_text=context_combined_text)

    # 3. Context Features
    ctx_feat = context_extractor.extract_features(context_turns, target_response=target_response)

    annotated_record = {
        "conversation_id": conversation_id,
        "target_message_id": target_message_id,
        "context": context_turns,
        "target_response": target_response,
        "messages": messages,
        "behavioral_metadata": behav_feat.to_dict(),
        "topic_metadata": topic_feat.to_dict(),
        "context_metadata": ctx_feat.to_dict(),
        "metadata": {
            **raw_metadata,
            "stage": 5,
            "has_behavioral_representation": True,
        },
    }
    return annotated_record


def build_ambiguous_prompts_benchmark(
    all_annotated_records: List[Dict[str, Any]],
    output_path: Path,
) -> List[Dict[str, Any]]:
    """
    Identifies short prompts with multiple distinct persona target responses.
    Prioritizes canonical prompts: Aaja, Khelega, Thik, Bsdk, Game.
    """
    priority_keywords = ["aaja", "khelega", "thik", "bsdk", "game"]

    # Group by normalized user prompt
    prompt_to_variants: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in all_annotated_records:
        context = r.get("context", [])
        if not context:
            continue

        # Look at the last user message
        last_user = ""
        for m in reversed(context):
            if m.get("role") == "user":
                last_user = m.get("content", "").strip()
                break

        if not last_user:
            continue

        norm_prompt = re.sub(r"[?!.,\s]+", " ", last_user).strip().title()

        # Track only if prompt is short (1-4 words)
        if len(norm_prompt.split()) <= 4:
            prompt_to_variants[norm_prompt].append({
                "conversation_id": r["conversation_id"],
                "target_message_id": r["target_message_id"],
                "raw_prompt": last_user,
                "context": r["context"],
                "target_response": r["target_response"],
                "primary_topic": r["topic_metadata"]["primary_topic"],
                "tone": r["behavioral_metadata"]["tone"],
                "response_type": r["behavioral_metadata"]["response_type"],
            })

    ambiguous_entries = []
    # Sort prompts: priority keywords first, then by number of unique responses descending
    def priority_score(prompt_name: str) -> int:
        p_low = prompt_name.lower()
        for idx, kw in enumerate(priority_keywords):
            if kw in p_low:
                return -100 + idx
        return 0

    sorted_prompts = sorted(
        prompt_to_variants.keys(),
        key=lambda p: (priority_score(p), -len(set(v["target_response"].strip().lower() for v in prompt_to_variants[p]))),
    )

    for p in sorted_prompts:
        variants = prompt_to_variants[p]
        unique_responses = set(v["target_response"].strip().lower() for v in variants)

        # Require at least 2 distinct target responses
        if len(unique_responses) >= 2:
            # Pick distinct examples (up to 10 variants per prompt)
            seen_resps = set()
            curated_variants = []
            for v in variants:
                r_key = v["target_response"].strip().lower()
                if r_key not in seen_resps:
                    seen_resps.add(r_key)
                    curated_variants.append(v)
                if len(curated_variants) >= 10:
                    break

            ambiguous_entries.append({
                "prompt": p,
                "variant_count": len(curated_variants),
                "total_occurrences": len(variants),
                "unique_target_responses": list(seen_resps),
                "variants": curated_variants,
            })

    save_jsonl(ambiguous_entries, output_path)
    return ambiguous_entries


def main():
    print("=" * 70)
    print("STAGE 5 — DATASET ENGINEERING & BEHAVIORAL REPRESENTATION")
    print("=" * 70)

    # Input paths (strictly read clean Stage 4 datasets)
    stage4_train_path = ml_root / "data/training/stage4_train.jsonl"
    clean_val_path = ml_root / "data/training/val.jsonl"
    clean_test_path = ml_root / "data/training/test.jsonl"

    assert stage4_train_path.is_file(), f"Missing {stage4_train_path}"
    assert clean_val_path.is_file(), f"Missing {clean_val_path}"
    assert clean_test_path.is_file(), f"Missing {clean_test_path}"

    # Output directory layout
    stage5_dir = ml_root / "data/stage5"
    annotated_dir = stage5_dir / "annotated"
    training_dir = stage5_dir / "training"
    reports_dir = stage5_dir / "reports"
    benchmarks_dir = stage5_dir / "benchmarks"

    annotated_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading clean Stage 4 datasets...")
    train_records = load_jsonl(stage4_train_path)
    val_records = load_jsonl(clean_val_path)
    test_records = load_jsonl(clean_test_path)

    print(f"  Stage 4 Train Records : {len(train_records):,}")
    print(f"  Validation Records    : {len(val_records):,}")
    print(f"  Test Records          : {len(test_records):,}")

    # Initialize extractors
    print(f"\n[2/4] Initializing feature extractors...")
    behavioral_extractor = BehavioralFeatureExtractor()
    topic_classifier = DeterministicTopicClassifier()
    context_extractor = ContextFeatureExtractor()

    print(f"\n[3/4] Annotating datasets with behavioral, topic, and context metadata...")
    train_annotated = [
        annotate_record(r, behavioral_extractor, topic_classifier, context_extractor)
        for r in train_records
    ]
    val_annotated = [
        annotate_record(r, behavioral_extractor, topic_classifier, context_extractor)
        for r in val_records
    ]
    test_annotated = [
        annotate_record(r, behavioral_extractor, topic_classifier, context_extractor)
        for r in test_records
    ]

    # Save to annotated/
    save_jsonl(train_annotated, annotated_dir / "train_annotated.jsonl")
    save_jsonl(val_annotated, annotated_dir / "val_annotated.jsonl")
    save_jsonl(test_annotated, annotated_dir / "test_annotated.jsonl")

    # Save to training/
    save_jsonl(train_annotated, training_dir / "stage5_train.jsonl")
    save_jsonl(val_annotated, training_dir / "val.jsonl")
    save_jsonl(test_annotated, training_dir / "test.jsonl")

    print(f"  Saved annotated datasets to: {annotated_dir}")
    print(f"  Saved Stage 5 training views to: {training_dir}")

    # Build ambiguous prompts benchmark
    print(f"\n[4/4] Generating ambiguous prompts benchmark...")
    ambiguous_path = benchmarks_dir / "ambiguous_prompts.jsonl"
    all_records = train_annotated + val_annotated + test_annotated
    ambiguous_entries = build_ambiguous_prompts_benchmark(all_records, ambiguous_path)
    print(f"  Identified {len(ambiguous_entries):,} ambiguous prompts with multiple target responses.")
    print(f"  Saved to: {ambiguous_path}")

    print("\nStage 5 dataset generation complete!")


if __name__ == "__main__":
    main()
