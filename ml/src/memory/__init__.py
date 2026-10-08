"""
Persistent Memory Layer for Persona Engine (Stage 6C).
Exports schemas, rule-based extraction, storage abstractions, ranking, and retrieval interfaces.
"""

from ml.src.memory.embedding import (
    BaseEmbeddingModel,
    DeterministicMockEmbedding,
    cosine_similarity,
)
from ml.src.memory.memory_extractor import MemoryExtractor
from ml.src.memory.memory_ranker import (
    MemoryRanker,
    MemoryScoreBreakdown,
    RankedMemory,
)
from ml.src.memory.memory_retriever import MemoryRetriever
from ml.src.memory.memory_schema import (
    ImportanceLevel,
    Memory,
    MemoryStatus,
    MemoryType,
)
from ml.src.memory.memory_store import (
    BaseMemoryStore,
    InMemoryMemoryStore,
    normalize_content_text,
)

__all__ = [
    "Memory",
    "MemoryType",
    "ImportanceLevel",
    "MemoryStatus",
    "MemoryExtractor",
    "BaseMemoryStore",
    "InMemoryMemoryStore",
    "normalize_content_text",
    "MemoryRanker",
    "RankedMemory",
    "MemoryScoreBreakdown",
    "MemoryRetriever",
    "BaseEmbeddingModel",
    "DeterministicMockEmbedding",
    "cosine_similarity",
]
