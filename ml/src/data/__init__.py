"""
Persona Engine — Data Subpackage
Handles dataset loading, validation, cleaning, and train/val/test splitting.
"""

from .cleaner import DataCleaner, clean_conversations
from .loader import (
    iter_jsonl,
    load_json,
    load_jsonl,
    normalize_conversation,
    save_json,
    save_jsonl,
)
from .splitter import DatasetSplitter, split_conversations
from .validator import DatasetValidator, ValidationResult, validate_dataset_file

__all__ = [
    "iter_jsonl",
    "load_json",
    "load_jsonl",
    "normalize_conversation",
    "save_json",
    "save_jsonl",
    "DatasetValidator",
    "ValidationResult",
    "validate_dataset_file",
    "DataCleaner",
    "clean_conversations",
    "DatasetSplitter",
    "split_conversations",
]
