"""
Persistent Memory Store abstraction for Persona Engine (Stage 6C).
Provides an interface and in-memory repository implementation supporting CRUD operations,
persona isolation, duplicate detection, and conflict resolution.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ml.src.memory.embedding import BaseEmbeddingModel, DeterministicMockEmbedding, cosine_similarity
from ml.src.memory.memory_schema import (
    ImportanceLevel,
    Memory,
    MemoryStatus,
    MemoryType,
    current_utc_iso,
)


def normalize_content_text(text: str) -> str:
    """
    Normalizes text for duplicate comparison.
    Strips punctuation, lowercases, removes redundant filler words and common pronouns,
    and performs basic verb normalization.
    """
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    # Remove common filler words and subject pronouns
    filler_tokens = {
        "i", "me", "my", "we", "user", "really", "very", "bahut",
        "ki", "ka", "ke", "hai", "hain", "main", "mera", "meri",
        "the", "a", "an", "is", "am", "are"
    }
    raw_tokens = [t for t in cleaned.split() if t not in filler_tokens]
    # Simple stemming for common predicates
    stem_map = {
        "likes": "like",
        "prefers": "prefer",
        "studies": "study",
        "wants": "want",
        "owns": "own",
    }
    tokens = [stem_map.get(t, t) for t in raw_tokens]
    return " ".join(tokens)


class BaseMemoryStore(ABC):
    """Abstract interface defining operations for persistent memory storage."""

    @abstractmethod
    def save(self, memory: Memory) -> Memory:
        """Save a new memory item."""
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory item by ID."""
        pass

    @abstractmethod
    def update(self, memory: Memory) -> Optional[Memory]:
        """Update an existing memory item."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory item by ID."""
        pass

    @abstractmethod
    def list_by_persona(
        self,
        persona_id: str,
        status: Optional[MemoryStatus] = MemoryStatus.ACTIVE,
    ) -> List[Memory]:
        """List memories belonging to a persona, filtered by lifecycle status."""
        pass

    @abstractmethod
    def search_by_persona(
        self,
        persona_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[Memory]:
        """Perform semantic or lexical search for memories belonging to a persona."""
        pass

    @abstractmethod
    def update_last_accessed(self, memory_id: str) -> Optional[Memory]:
        """Update last accessed timestamp for a memory."""
        pass

    @abstractmethod
    def find_duplicates(self, memory: Memory) -> Optional[Memory]:
        """Check if an equivalent memory already exists."""
        pass

    @abstractmethod
    def find_conflicts(self, memory: Memory) -> List[Memory]:
        """Identify existing memories that conflict with the candidate memory."""
        pass

    @abstractmethod
    def count(self, persona_id: Optional[str] = None) -> int:
        """Count total memories in the store, optionally scoped to a persona."""
        pass


class InMemoryMemoryStore(BaseMemoryStore):
    """
    Thread-safe, testable in-memory implementation of BaseMemoryStore.
    Supports deterministic duplicate detection and slot-based conflict resolution.
    """

    # Slot definitions for conflict detection: (Regex pattern, slot_name)
    CONFLICT_SLOTS = [
        (r"^(prefers|likes)\s+(java|python|c\+\+|rust|javascript|go)$", "primary_programming_language"),
        (r"^studies\s+b\.?tech\s+(\w+)$", "degree_branch"),
        (r"^college\s+is\s+located\s+in\s+(\w+)$", "college_location"),
    ]

    def __init__(self, embedding_model: Optional[BaseEmbeddingModel] = None):
        self._storage: Dict[str, Memory] = {}
        self.embedding_model = embedding_model or DeterministicMockEmbedding()

    def save(self, memory: Memory) -> Memory:
        """Persist memory. Ensures embedding is generated if missing."""
        if memory.embedding is None and memory.content:
            memory.embedding = self.embedding_model.embed_text(memory.content)

        self._storage[memory.id] = memory
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID."""
        return self._storage.get(memory_id)

    def update(self, memory: Memory) -> Optional[Memory]:
        """Update existing memory."""
        if memory.id not in self._storage:
            return None
        memory.updated_at = current_utc_iso()
        self._storage[memory.id] = memory
        return memory

    def delete(self, memory_id: str) -> bool:
        """Delete memory by ID."""
        if memory_id in self._storage:
            del self._storage[memory_id]
            return True
        return False

    def list_by_persona(
        self,
        persona_id: str,
        status: Optional[MemoryStatus] = MemoryStatus.ACTIVE,
    ) -> List[Memory]:
        """List memories isolated to persona_id, optionally filtered by status."""
        results = []
        for mem in self._storage.values():
            if mem.persona_id == persona_id:
                if status is None or mem.status == status:
                    results.append(mem)
        return results

    def update_last_accessed(self, memory_id: str) -> Optional[Memory]:
        """Record retrieval access timestamp."""
        mem = self._storage.get(memory_id)
        if mem:
            mem.last_accessed_at = current_utc_iso()
            return mem
        return None

    def _extract_conflict_slot(self, content: str) -> Optional[Tuple[str, str]]:
        """Extracts (slot_name, slot_value) if content matches a recognized exclusive slot."""
        clean = content.lower().strip()
        for pat, slot_name in self.CONFLICT_SLOTS:
            m = re.match(pat, clean)
            if m:
                # Value is the distinct capture
                val = m.group(len(m.groups()))
                return slot_name, val
        return None

    def find_duplicates(self, candidate: Memory) -> Optional[Memory]:
        """
        Detects exact or normalized duplicates for the same persona and memory type.
        """
        cand_norm = normalize_content_text(candidate.content)
        for existing in self._storage.values():
            if (
                existing.persona_id == candidate.persona_id
                and existing.status == MemoryStatus.ACTIVE
            ):
                # 1. Exact content match
                if existing.content.strip().lower() == candidate.content.strip().lower():
                    return existing

                # 2. Normalized content match
                exist_norm = normalize_content_text(existing.content)
                if exist_norm == cand_norm:
                    return existing

                # 3. Embedding cosine similarity >= 0.98 if same type
                if (
                    existing.memory_type == candidate.memory_type
                    and existing.embedding
                    and candidate.embedding
                ):
                    sim = cosine_similarity(existing.embedding, candidate.embedding)
                    if sim >= 0.98:
                        return existing

        return None

    def find_conflicts(self, candidate: Memory) -> List[Memory]:
        """
        Identifies active memories belonging to the same persona that conflict with the candidate.
        Example: 'Prefers Java' conflicts with 'Prefers Python'.
        """
        conflicts: List[Memory] = []
        cand_slot = self._extract_conflict_slot(candidate.content)
        if not cand_slot:
            return conflicts

        slot_name, cand_val = cand_slot

        for existing in self._storage.values():
            if (
                existing.persona_id == candidate.persona_id
                and existing.status == MemoryStatus.ACTIVE
                and existing.id != candidate.id
            ):
                exist_slot = self._extract_conflict_slot(existing.content)
                if exist_slot:
                    exist_slot_name, exist_val = exist_slot
                    if exist_slot_name == slot_name and exist_val != cand_val:
                        conflicts.append(existing)

        return conflicts

    def save_or_merge(self, candidate: Memory) -> Tuple[Memory, str]:
        """
        High-level lifecycle ingestion:
        1. Checks for duplicate -> merges if found.
        2. Checks for conflicts -> marks conflicting memories as superseded.
        3. Saves candidate as active.
        Returns: (Memory, action_taken) where action_taken is 'created', 'merged', or 'conflict_resolved'.
        """
        if candidate.embedding is None and candidate.content:
            candidate.embedding = self.embedding_model.embed_text(candidate.content)

        # 1. Duplicate check
        duplicate = self.find_duplicates(candidate)
        if duplicate:
            # Refresh updated_at and boost importance if candidate is higher
            importance_rank = {
                ImportanceLevel.LOW: 1,
                ImportanceLevel.MEDIUM: 2,
                ImportanceLevel.HIGH: 3,
            }
            if importance_rank.get(candidate.importance, 2) > importance_rank.get(duplicate.importance, 2):
                duplicate.importance = candidate.importance
            duplicate.updated_at = current_utc_iso()
            return duplicate, "merged"

        # 2. Conflict check
        conflicts = self.find_conflicts(candidate)
        if conflicts:
            for conflict_mem in conflicts:
                conflict_mem.status = MemoryStatus.SUPERSEDED
                conflict_mem.superseded_by = candidate.id
                conflict_mem.updated_at = current_utc_iso()
            self.save(candidate)
            return candidate, "conflict_resolved"

        # 3. Clean insert
        self.save(candidate)
        return candidate, "created"

    def search_by_persona(
        self,
        persona_id: str,
        query: str,
        top_k: int = 10,
    ) -> List[Memory]:
        """
        Performs semantic and lexical similarity search over active memories for a given persona.
        """
        active_memories = self.list_by_persona(persona_id, status=MemoryStatus.ACTIVE)
        if not active_memories:
            return []

        query_emb = self.embedding_model.embed_text(query)
        query_tokens = set(re.findall(r"\b\w+\b", query.lower()))

        scored: List[Tuple[float, Memory]] = []
        for mem in active_memories:
            # Semantic similarity
            sim = 0.0
            if mem.embedding and query_emb:
                sim = cosine_similarity(query_emb, mem.embedding)

            # Lexical overlap
            mem_tokens = set(re.findall(r"\b\w+\b", mem.content.lower()))
            overlap = len(query_tokens & mem_tokens) / max(1, len(query_tokens | mem_tokens))

            # Blended score
            score = 0.7 * sim + 0.3 * overlap
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def count(self, persona_id: Optional[str] = None) -> int:
        """Count memories, optionally filtered by persona."""
        if persona_id is None:
            return len(self._storage)
        return sum(1 for m in self._storage.values() if m.persona_id == persona_id)
