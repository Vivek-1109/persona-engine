"""
Vector embedding abstraction for Persona Engine Memory Layer (Stage 6C).
Provides an interface for dense vector representation with deterministic mock fallback
to ensure unit testing without downloading multi-gigabyte models.
"""

from abc import ABC, abstractmethod
import math
import re
from typing import List, Optional


class BaseEmbeddingModel(ABC):
    """Abstract interface for generating dense vector embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a float vector."""
        pass

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings."""
        return [self.embed_text(t) for t in texts]


class DeterministicMockEmbedding(BaseEmbeddingModel):
    """
    Deterministic pseudo-semantic embedding generator.
    Creates normalized 64-dimensional float vectors derived from token hashes.
    Guarantees deterministic results without external model downloads or network access.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        clean = text.lower().strip()
        tokens = re.findall(r"\b\w+\b", clean)
        if not tokens:
            return vec

        for token in tokens:
            # Deterministic hash scattering
            token_hash = hash(token) % 1000003
            idx = token_hash % self.dimension
            val = (token_hash % 97) / 100.0 + 0.1
            vec[idx] += val

        # L2 normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))
