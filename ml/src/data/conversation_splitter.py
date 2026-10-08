#!/usr/bin/env python3
"""
Persona Engine — Conversation-Level Splitter
Splits dataset examples strictly at the conversation level (by conversation_id)
to prevent data leakage across train, validation, and test sets.
Uses deterministic seeding and verifies zero conversation overlap.
"""

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Set, Tuple


@dataclass
class SplitStats:
    total_conversations: int
    train_conversations: int
    val_conversations: int
    test_conversations: int
    total_examples: int
    train_examples: int
    val_examples: int
    test_examples: int
    is_leak_free: bool


class ConversationSplitter:
    """
    Performs leakage-free conversation-level partitioning.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        diff = abs((train_ratio + val_ratio + test_ratio) - 1.0)
        if diff > 1e-4:
            raise ValueError(f"Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio})")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self, examples: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], SplitStats]:
        """
        Partition examples into train, validation, and test subsets based on conversation_id.
        """
        # Collect unique conversation IDs deterministically
        unique_conv_ids = sorted(list({ex["metadata"]["conversation_id"] for ex in examples}))
        n_total = len(unique_conv_ids)

        if n_total == 0:
            empty_stats = SplitStats(0, 0, 0, 0, 0, 0, 0, 0, True)
            return [], [], [], empty_stats

        # Deterministic shuffle
        rng = random.Random(self.seed)
        shuffled_convs = list(unique_conv_ids)
        rng.shuffle(shuffled_convs)

        n_train = int(round(n_total * self.train_ratio))
        n_val = int(round(n_total * self.val_ratio))
        # Ensure at least 1 conversation in val/test if enough exist
        if n_total >= 3:
            n_train = max(1, min(n_train, n_total - 2))
            n_val = max(1, min(n_val, n_total - n_train - 1))

        train_conv_set: Set[str] = set(shuffled_convs[:n_train])
        val_conv_set: Set[str] = set(shuffled_convs[n_train:n_train + n_val])
        test_conv_set: Set[str] = set(shuffled_convs[n_train + n_val:])

        # Strict leakage checks
        overlap_train_val = train_conv_set.intersection(val_conv_set)
        overlap_train_test = train_conv_set.intersection(test_conv_set)
        overlap_val_test = val_conv_set.intersection(test_conv_set)

        if overlap_train_val or overlap_train_test or overlap_val_test:
            raise ValueError(
                f"Data leakage detected! "
                f"train∩val: {len(overlap_train_val)}, "
                f"train∩test: {len(overlap_train_test)}, "
                f"val∩test: {len(overlap_val_test)}"
            )

        train_examples = [ex for ex in examples if ex["metadata"]["conversation_id"] in train_conv_set]
        val_examples = [ex for ex in examples if ex["metadata"]["conversation_id"] in val_conv_set]
        test_examples = [ex for ex in examples if ex["metadata"]["conversation_id"] in test_conv_set]

        # Double check example level
        train_ex_convs = {ex["metadata"]["conversation_id"] for ex in train_examples}
        val_ex_convs = {ex["metadata"]["conversation_id"] for ex in val_examples}
        test_ex_convs = {ex["metadata"]["conversation_id"] for ex in test_examples}

        is_leak_free = bool(
            len(train_ex_convs & val_ex_convs) == 0
            and len(train_ex_convs & test_ex_convs) == 0
            and len(val_ex_convs & test_ex_convs) == 0
        )

        stats = SplitStats(
            total_conversations=n_total,
            train_conversations=len(train_conv_set),
            val_conversations=len(val_conv_set),
            test_conversations=len(test_conv_set),
            total_examples=len(examples),
            train_examples=len(train_examples),
            val_examples=len(val_examples),
            test_examples=len(test_examples),
            is_leak_free=is_leak_free,
        )

        return train_examples, val_examples, test_examples, stats
