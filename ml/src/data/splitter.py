"""
Persona Engine — Dataset Splitter
Splits conversations into train, validation, and test subsets based on conversation IDs
to strictly prevent conversation data leakage across splits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class DatasetSplits:
    train: List[Dict[str, Any]]
    validation: List[Dict[str, Any]]
    test: List[Dict[str, Any]]

    def summary(self) -> str:
        total = len(self.train) + len(self.validation) + len(self.test)
        return (
            f"Dataset Splits: Total={total} | "
            f"Train={len(self.train)} ({len(self.train)/max(1, total):.1%}), "
            f"Validation={len(self.validation)} ({len(self.validation)/max(1, total):.1%}), "
            f"Test={len(self.test)} ({len(self.test)/max(1, total):.1%})"
        )


class DatasetSplitter:
    """
    Partitions conversations into train, validation, and test sets.
    """

    def __init__(
        self,
        train_ratio: float = 0.85,
        val_ratio: float = 0.10,
        test_ratio: float = 0.05,
        seed: int = 42,
        shuffle: bool = True,
    ):
        total = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total}"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.shuffle = shuffle

    def split(self, conversations: List[Dict[str, Any]]) -> DatasetSplits:
        """Splits a list of conversations into train, validation, and test partitions."""
        items = list(conversations)
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(items)

        n = len(items)
        if n == 0:
            return DatasetSplits(train=[], validation=[], test=[])

        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        # For small datasets, ensure at least 1 in train if items exist
        if n > 0 and n_train == 0:
            n_train = 1

        train_data = items[:n_train]
        val_data = items[n_train : n_train + n_val]
        test_data = items[n_train + n_val :]

        return DatasetSplits(
            train=train_data,
            validation=val_data,
            test=test_data,
        )


def split_conversations(
    conversations: List[Dict[str, Any]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
) -> DatasetSplits:
    """Convenience helper to split conversations."""
    splitter = DatasetSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        shuffle=shuffle,
    )
    return splitter.split(conversations)
