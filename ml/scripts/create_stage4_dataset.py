#!/usr/bin/env python3
"""
Persona Engine — Stage 4 Dataset View Builder
Generates ml/data/training/stage4_train.jsonl from clean train.jsonl.
Applies:
- Conversation cap: maximum 75 examples per conversation
- Context turn sampling:
  - 1-turn: 40%
  - 2-turn: 25%
  - 4-turn: 20%
  - 6-turn: 15%
- Deterministic seed: 42

Guarantees zero data leakage and strict separation from val.jsonl and test.jsonl.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

# Fix Windows console UTF-8 output if necessary
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml root in sys.path
ml_root = Path(__file__).resolve().parents[1]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if clean:
                records.append(json.loads(clean))
    return records


def save_jsonl(records: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_stage4_dataset():
    training_dir = ml_root / "data" / "training"
    train_path = training_dir / "train.jsonl"
    val_path = training_dir / "val.jsonl"
    test_path = training_dir / "test.jsonl"
    stage4_train_path = training_dir / "stage4_train.jsonl"

    print(f"[STAGE 4] Loading source training dataset from: {train_path}")
    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)
    test_records = load_jsonl(test_path)

    orig_total = len(train_records)
    print(f"Original train examples: {orig_total:,}")

    # 1. Measure context distribution before sampling
    ctx_before = Counter(ex["metadata"]["context_turns"] for ex in train_records)
    print("Context distribution before sampling:")
    for k in [1, 2, 4, 6]:
        cnt = ctx_before[k]
        pct = (cnt / orig_total) * 100
        print(f"  {k}-turn: {cnt:,} ({pct:.2f}%)")

    # Measure conversation distribution before cap
    conv_before = Counter(ex["metadata"]["conversation_id"] for ex in train_records)
    counts_before = list(conv_before.values())
    print(f"Examples per conversation before cap: min={min(counts_before)}, max={max(counts_before)}, mean={sum(counts_before)/len(counts_before):.2f}")

    # 2. Apply Conversation Cap of 75
    conv_to_ex = defaultdict(list)
    for ex in train_records:
        conv_to_ex[ex["metadata"]["conversation_id"]].append(ex)

    capped_records = []
    dropped_by_cap = 0
    max_cap = 75

    # Deterministic sorting of conversations
    for cid in sorted(conv_to_ex.keys()):
        exs = conv_to_ex[cid]
        if len(exs) > max_cap:
            dropped_by_cap += (len(exs) - max_cap)
            capped_records.extend(exs[:max_cap])
        else:
            capped_records.extend(exs)

    print(f"\nExamples removed by conversation cap (max {max_cap}): {dropped_by_cap}")
    print(f"Total examples remaining after cap: {len(capped_records):,}")

    conv_after_cap = Counter(ex["metadata"]["conversation_id"] for ex in capped_records)
    counts_after_cap = list(conv_after_cap.values())
    print(f"Examples per conversation after cap: min={min(counts_after_cap)}, max={max(counts_after_cap)}, mean={sum(counts_after_cap)/len(counts_after_cap):.2f}")

    # 3. Context Turn Stratified Sampling: 40% (1-turn), 25% (2-turn), 20% (4-turn), 15% (6-turn)
    by_depth = defaultdict(list)
    for ex in capped_records:
        by_depth[ex["metadata"]["context_turns"]].append(ex)

    rng = random.Random(42)

    # Calculate optimal target sample size based on rarest available depth (6-turn: 15%)
    # max possible N = count(6-turn) / 0.15
    count_6 = len(by_depth[6])
    target_n = int(count_6 / 0.15)  # e.g., 564 / 0.15 = 3760

    target_counts = {
        1: int(round(target_n * 0.40)),
        2: int(round(target_n * 0.25)),
        4: int(round(target_n * 0.20)),
        6: int(round(target_n * 0.15)),
    }

    # Ensure total sums exactly to target_n
    diff = target_n - sum(target_counts.values())
    target_counts[1] += diff

    print(f"\nTarget sample sizes: {target_counts} (Total: {target_n:,})")

    sampled_records = []
    for k in [1, 2, 4, 6]:
        pool = by_depth[k]
        if len(pool) < target_counts[k]:
            raise ValueError(f"Pool for depth {k} has only {len(pool)} items, but need {target_counts[k]}")
        # Deterministic sampling
        sampled_k = rng.sample(pool, target_counts[k])
        sampled_records.extend(sampled_k)

    # Sort deterministically by conversation, target message, and context depth
    sampled_records.sort(
        key=lambda x: (
            x["metadata"]["conversation_id"],
            x["metadata"]["target_message_id"],
            x["metadata"]["context_turns"],
        )
    )

    stage4_total = len(sampled_records)
    print(f"\nStage-4 train examples: {stage4_total:,}")

    ctx_after = Counter(ex["metadata"]["context_turns"] for ex in sampled_records)
    print("Context distribution after sampling:")
    for k in [1, 2, 4, 6]:
        cnt = ctx_after[k]
        pct = (cnt / stage4_total) * 100
        print(f"  {k}-turn: {cnt:,} ({pct:.2f}%)")

    # 4. Strict Data Integrity Assertions
    train_ids = {ex["metadata"]["example_id"] for ex in train_records}
    val_ids = {ex["metadata"]["example_id"] for ex in val_records}
    test_ids = {ex["metadata"]["example_id"] for ex in test_records}

    stage4_ids = {ex["metadata"]["example_id"] for ex in sampled_records}

    # Verify origin
    assert stage4_ids.issubset(train_ids), "Data error: Stage-4 contains examples not in train.jsonl!"
    assert len(stage4_ids & val_ids) == 0, "Data leakage: Stage-4 contains validation examples!"
    assert len(stage4_ids & test_ids) == 0, "Data leakage: Stage-4 contains test examples!"

    # Verify conversation disjointness
    train_convs = {ex["metadata"]["conversation_id"] for ex in sampled_records}
    val_convs = {ex["metadata"]["conversation_id"] for ex in val_records}
    test_convs = {ex["metadata"]["conversation_id"] for ex in test_records}

    assert len(train_convs & val_convs) == 0, "Conversation leakage: Stage-4 shares conversations with val!"
    assert len(train_convs & test_convs) == 0, "Conversation leakage: Stage-4 shares conversations with test!"

    # Verify content and target role
    for ex in sampled_records:
        msgs = ex["messages"]
        assert msgs[-1]["role"] == "assistant", "Target role must be assistant!"
        for m in msgs:
            assert "waiting for this message" not in m["content"].lower(), "Protocol artifact detected!"
            assert '"metadata":' not in m["content"], "Accidental metadata leakage in content!"

    print("\n[INTEGRITY] All assertions passed: 100% origin match, 0 leakage, 0 artifacts, valid roles.")

    # 5. Save stage4_train.jsonl
    save_jsonl(sampled_records, stage4_train_path)
    print(f"[STAGE 4] Saved Stage-4 training dataset to: {stage4_train_path}")


if __name__ == "__main__":
    build_stage4_dataset()
