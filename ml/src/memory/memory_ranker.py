"""
Deterministic, explainable ranking engine for Persona Engine Memory Layer (Stage 6C).
Computes multi-factor relevance scores combining semantic alignment, topic match,
importance level, and temporal recency.
"""

from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ml.src.memory.embedding import cosine_similarity
from ml.src.memory.memory_schema import ImportanceLevel, Memory


class MemoryScoreBreakdown(BaseModel):
    """Detailed, explainable score breakdown for a candidate memory."""
    final_score: float = Field(..., ge=0.0, le=1.0, description="Total composite ranking score.")
    semantic_score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity contribution.")
    topic_score: float = Field(..., ge=0.0, le=1.0, description="Conversational topic match contribution.")
    importance_score: float = Field(..., ge=0.0, le=1.0, description="Importance level contribution.")
    recency_score: float = Field(..., ge=0.0, le=1.0, description="Temporal recency contribution.")
    explanation: str = Field(..., description="Human-readable breakdown explanation.")


class RankedMemory(BaseModel):
    """A memory item paired with its calculated ranking score and breakdown."""
    memory: Memory
    score: float
    breakdown: MemoryScoreBreakdown


class MemoryRanker:
    """
    Ranks memories using an explainable weighted linear combination:
    Score = (W_semantic * semantic_score) +
            (W_topic * topic_score) +
            (W_importance * importance_score) +
            (W_recency * recency_score)
    """

    DEFAULT_SEMANTIC_WEIGHT = 0.40
    DEFAULT_TOPIC_WEIGHT = 0.30
    DEFAULT_IMPORTANCE_WEIGHT = 0.20
    DEFAULT_RECENCY_WEIGHT = 0.10

    # Keyword associations for domain topic matching
    TOPIC_KEYWORD_MAP = {
        "gaming": {"game", "gaming", "bgmi", "valorant", "steam", "tournament", "khelega", "khelte", "match"},
        "college": {"college", "dean", "attendance", "btech", "cse", "semester", "exam", "sih", "notice", "hostel", "roommate"},
        "technology": {"laptop", "battery", "drain", "code", "coding", "java", "python", "spring", "backend", "developer", "hardware"},
        "movies": {"movie", "film", "marvel", "cinema", "series", "watch", "avengers"},
        "plans": {"kal", "tomorrow", "visit", "trip", "milte", "plan"},
        "social": {"friend", "sister", "bhai", "family", "dost"},
    }

    def __init__(
        self,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        topic_weight: float = DEFAULT_TOPIC_WEIGHT,
        importance_weight: float = DEFAULT_IMPORTANCE_WEIGHT,
        recency_weight: float = DEFAULT_RECENCY_WEIGHT,
    ):
        self.semantic_weight = semantic_weight
        self.topic_weight = topic_weight
        self.importance_weight = importance_weight
        self.recency_weight = recency_weight

    def _compute_semantic_score(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        memory: Memory,
    ) -> float:
        """Computes semantic similarity via embedding cosine similarity or token overlap."""
        if query_embedding and memory.embedding:
            return cosine_similarity(query_embedding, memory.embedding)

        # Fallback to token overlap
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        m_tokens = set(re.findall(r"\b\w+\b", memory.content.lower()))
        if not q_tokens or not m_tokens:
            return 0.0

        intersection = len(q_tokens & m_tokens)
        union = len(q_tokens | m_tokens)
        return float(intersection / union) if union > 0 else 0.0

    def _compute_topic_score(
        self,
        memory: Memory,
        context_topic: Optional[str],
        query: str,
    ) -> float:
        """
        Computes topic alignment score (0.0 to 1.0).
        1.0 if memory topic matches context_topic.
        1.0 if query tokens strongly trigger memory topic keywords.
        0.0 otherwise.
        """
        mem_topic = (memory.topic or "").lower()
        if not mem_topic:
            return 0.0

        # Direct match with context topic
        if context_topic and context_topic.lower() == mem_topic:
            return 1.0

        # Query token keyword trigger
        q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
        domain_keywords = self.TOPIC_KEYWORD_MAP.get(mem_topic, set())
        if q_tokens & domain_keywords:
            return 1.0

        return 0.0

    def _compute_importance_score(self, memory: Memory) -> float:
        """Maps categorical importance level to normalized float weight."""
        mapping = {
            ImportanceLevel.HIGH: 1.0,
            ImportanceLevel.MEDIUM: 0.6,
            ImportanceLevel.LOW: 0.2,
        }
        return mapping.get(memory.importance, 0.6)

    def _compute_recency_score(self, memory: Memory, reference_time: Optional[datetime] = None) -> float:
        """
        Computes recency score using exponential decay over days.
        Recent memories (< 1 day) receive ~1.0. Memories older than 30 days decay gracefully.
        """
        ref = reference_time or datetime.now(timezone.utc)
        ts_str = memory.updated_at or memory.created_at
        try:
            mem_time = datetime.fromisoformat(ts_str)
            if mem_time.tzinfo is None:
                mem_time = mem_time.replace(tzinfo=timezone.utc)
            delta_days = max(0.0, (ref - mem_time).total_seconds() / 86400.0)
            # Half-life of 30 days
            return float(math.exp(-delta_days / 30.0))
        except Exception:
            return 0.5

    def score_memory(
        self,
        memory: Memory,
        query: str,
        query_embedding: Optional[List[float]] = None,
        context_topic: Optional[str] = None,
        reference_time: Optional[datetime] = None,
    ) -> RankedMemory:
        """Calculates composite ranking score and explainability breakdown."""
        semantic_score = self._compute_semantic_score(query, query_embedding, memory)
        topic_score = self._compute_topic_score(memory, context_topic, query)
        importance_score = self._compute_importance_score(memory)
        recency_score = self._compute_recency_score(memory, reference_time)

        final_score = (
            self.semantic_weight * semantic_score
            + self.topic_weight * topic_score
            + self.importance_weight * importance_score
            + self.recency_weight * recency_score
        )
        final_score = max(0.0, min(1.0, final_score))

        explanation = (
            f"Composite={final_score:.3f} | "
            f"Semantic({self.semantic_weight:.2f})={semantic_score:.3f}, "
            f"Topic({self.topic_weight:.2f})={topic_score:.3f}, "
            f"Importance({self.importance_weight:.2f})={importance_score:.3f}, "
            f"Recency({self.recency_weight:.2f})={recency_score:.3f}"
        )

        breakdown = MemoryScoreBreakdown(
            final_score=round(final_score, 4),
            semantic_score=round(semantic_score, 4),
            topic_score=round(topic_score, 4),
            importance_score=round(importance_score, 4),
            recency_score=round(recency_score, 4),
            explanation=explanation,
        )

        return RankedMemory(memory=memory, score=round(final_score, 4), breakdown=breakdown)

    def rank_memories(
        self,
        memories: List[Memory],
        query: str,
        query_embedding: Optional[List[float]] = None,
        context_topic: Optional[str] = None,
        top_k: int = 5,
    ) -> List[RankedMemory]:
        """Scores and ranks a list of candidate memories in descending order of score."""
        scored = [
            self.score_memory(
                memory=m,
                query=query,
                query_embedding=query_embedding,
                context_topic=context_topic,
            )
            for m in memories
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
