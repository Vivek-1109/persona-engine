"""
Strongly-typed schemas and taxonomy enums for the Persona Engine Memory Layer (Stage 6C).
Defines persistent Memory models, importance levels, memory types, and lifecycle states.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Controlled taxonomy of persistent memory categories."""
    FACT = "fact"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"
    RELATIONSHIP = "relationship"
    PLAN = "plan"
    EVENT = "event"
    GOAL = "goal"


class ImportanceLevel(str, Enum):
    """Deterministic significance rating for memory retention and retrieval weighting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MemoryStatus(str, Enum):
    """Lifecycle status for duplicate resolution and conflict tracking."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


def current_utc_iso() -> str:
    """Helper returning ISO-8601 formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class Memory(BaseModel):
    """
    Core persistent memory unit representing an extracted knowledge item about
    a user or persona entity across conversational sessions.
    """
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique identifier for this memory."
    )
    persona_id: str = Field(
        default="vivek",
        description="Identifier of the persona this memory belongs to or interacts with."
    )
    memory_type: MemoryType = Field(
        ...,
        description="Taxonomic classification of the memory item."
    )
    content: str = Field(
        ...,
        description="Factual, concise canonical representation of the memory."
    )
    importance: ImportanceLevel = Field(
        default=ImportanceLevel.MEDIUM,
        description="Importance weight affecting retrieval ranking."
    )
    topic: Optional[str] = Field(
        default=None,
        description="Associated topic domain (e.g. gaming, college, technology) for alignment."
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Dense vector embedding representation for semantic retrieval."
    )
    source_conversation_id: Optional[str] = Field(
        default=None,
        description="Identifier of the conversation where this memory was uttered."
    )
    source_message_id: Optional[Union[str, int]] = Field(
        default=None,
        description="Turn index or message ID that yielded this memory."
    )
    status: MemoryStatus = Field(
        default=MemoryStatus.ACTIVE,
        description="Active state or superseded by newer conflicting knowledge."
    )
    superseded_by: Optional[str] = Field(
        default=None,
        description="Memory ID of newer conflicting fact if superseded."
    )
    created_at: str = Field(
        default_factory=current_utc_iso,
        description="Creation timestamp in ISO-8601 UTC."
    )
    updated_at: str = Field(
        default_factory=current_utc_iso,
        description="Last updated timestamp in ISO-8601 UTC."
    )
    last_accessed_at: Optional[str] = Field(
        default=None,
        description="Timestamp when this memory was last retrieved into prompt context."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to JSON-serializable dictionary."""
        return self.model_dump()
