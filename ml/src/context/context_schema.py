"""
Strongly-typed schemas and taxonomy enums for the Persona Engine Context Engine.
Defines ContextAnalysis output and all categorical dimensions.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TopicCategory(str, Enum):
    """Controlled topic taxonomy matching Stage 5 standards."""
    CASUAL_CHAT = "casual_chat"
    PLANS = "plans"
    TECHNOLOGY = "technology"
    GAMING = "gaming"
    SOCIAL = "social"
    COLLEGE = "college"
    SPORTS = "sports"
    MOVIES = "movies"
    OTHER = "other"


class ConversationState(str, Enum):
    """Runtime dialogue progression states."""
    OPENING = "opening"
    ONGOING = "ongoing"
    INQUIRY = "inquiry"
    BANTER = "banter"
    AGREEMENT = "agreement"
    CLOSING = "closing"


class UserIntent(str, Enum):
    """Controlled user dialogue act intent taxonomy."""
    QUESTION = "question"
    ANSWER_REQUEST = "answer_request"
    INVITATION = "invitation"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    INFORMATION = "information"
    REACTION = "reaction"
    PLANNING = "planning"
    CASUAL_CHAT = "casual_chat"
    UNKNOWN = "unknown"


class AmbiguityLevel(str, Enum):
    """Degree to which the final message relies on conversational context."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContextAnalysis(BaseModel):
    """
    Structured context intelligence describing current conversation dynamics.
    Consumed upstream by the orchestrator and dynamic prompt assembly.
    """
    topic: TopicCategory = Field(
        ...,
        description="Inferred conversational domain across all recent turns."
    )
    conversation_state: ConversationState = Field(
        ...,
        description="Current structural dialogue state."
    )
    user_intent: UserIntent = Field(
        ...,
        description="Inferred user dialogue act of the most recent user turn."
    )
    ambiguity: AmbiguityLevel = Field(
        ...,
        description="Context dependence level of the current user message."
    )
    recent_activity: Optional[str] = Field(
        default=None,
        description="Concise description of recent conversational activity, or None."
    )
    context_depth: int = Field(
        ...,
        ge=1,
        description="Number of messages provided in recent context window."
    )
    last_user_message: str = Field(
        ...,
        description="Cleaned text of the latest user utterance."
    )
    speaker_alternation_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Frequency of speaker transitions between turns (0.0 to 1.0)."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to JSON-serializable dictionary."""
        return self.model_dump()
