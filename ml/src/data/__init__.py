"""
Persona Engine — Data Subpackage
Handles WhatsApp parsing, conversation segmentation, behavioral annotation,
candidate extraction, quality filtering, and leakage-free dataset splitting.
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
from .whatsapp_parser import WhatsAppParser, ParsedMessage
from .conversation_segmenter import ConversationSegmenter, ConversationSession, DialogueTurn
from .behavioral_annotator import BehavioralAnnotator, BehavioralAnnotation
from .candidate_extractor import CandidateExtractor
from .conversation_splitter import ConversationSplitter, SplitStats

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
    "WhatsAppParser",
    "ParsedMessage",
    "ConversationSegmenter",
    "ConversationSession",
    "DialogueTurn",
    "BehavioralAnnotator",
    "BehavioralAnnotation",
    "CandidateExtractor",
    "ConversationSplitter",
    "SplitStats",
]
