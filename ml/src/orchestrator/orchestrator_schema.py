"""
Strongly-typed schemas and taxonomy enums for the Persona Engine Conversation Orchestrator (Stage 6D).
Defines ConversationRequest, ResponsePlan, ResponseStrategy, and ResponseTone.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from ml.src.context.context_schema import (
    AmbiguityLevel,
    ContextAnalysis,
    ConversationState,
    TopicCategory,
    UserIntent,
)
from ml.src.memory.memory_ranker import RankedMemory
from ml.src.memory.memory_schema import Memory


class ResponseStrategy(str, Enum):
    """Controlled taxonomy of dialogue action strategies."""
    ANSWER = "answer"
    ACKNOWLEDGE = "acknowledge"
    ASK_CLARIFICATION = "ask_clarification"
    ACCEPT = "accept"
    DECLINE = "decline"
    SUGGEST = "suggest"
    REACT = "react"
    CONTINUE_BANTER = "continue_banter"
    PROVIDE_INFORMATION = "provide_information"
    CLOSE_CONVERSATION = "close_conversation"


class ResponseTone(str, Enum):
    """High-level communicative tone guidance for response generation."""
    CASUAL = "casual"
    NEUTRAL = "neutral"
    SERIOUS = "serious"
    HUMOROUS = "humorous"
    SUPPORTIVE = "supportive"
    TEASING = "teasing"


class ConversationRequest(BaseModel):
    """
    Input payload provided to ConversationOrchestrator.
    Consolidates dialogue history with upstream intelligence from Context & Memory Engines.
    """
    persona_id: str = Field(
        default="vivek",
        description="Target persona identifier."
    )
    conversation_id: str = Field(
        default="conv_default",
        description="Conversation session identifier."
    )
    messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Chronological dialogue history [{'role': 'user'|'assistant', 'content': '...'}]."
    )
    context: Optional[ContextAnalysis] = Field(
        default=None,
        description="Stage 6B ContextAnalysis output (if precomputed)."
    )
    memories: Optional[List[Union[RankedMemory, Memory]]] = Field(
        default=None,
        description="Stage 6C candidate memories retrieved for this conversation."
    )
    previous_response: Optional[str] = Field(
        default=None,
        description="Last generated response from the persona, if available."
    )
    persona_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional persona configuration or metadata."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return self.model_dump()


class ResponsePlan(BaseModel):
    """
    Deterministic decision plan output by the Orchestrator.
    Instructs Stage 6A generation on strategy, tone, relevant memory, and constraints.
    """
    topic: TopicCategory = Field(
        ...,
        description="Topic domain governing the turn."
    )
    user_intent: UserIntent = Field(
        ...,
        description="Inferred user dialogue intent."
    )
    conversation_state: ConversationState = Field(
        ...,
        description="Current conversational state."
    )
    response_strategy: ResponseStrategy = Field(
        ...,
        description="High-level dialogue action strategy to execute."
    )
    tone: ResponseTone = Field(
        ...,
        description="Target high-level tone guidance."
    )
    use_memory: bool = Field(
        default=False,
        description="Whether relevant persistent memory should inform generation."
    )
    selected_memory_ids: List[str] = Field(
        default_factory=list,
        description="IDs of vetted memories approved for inclusion in the prompt context."
    )
    selected_memories: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary representation of selected memories for generation payload."
    )
    context_depth: int = Field(
        default=1,
        ge=0,
        description="Number of context messages evaluated."
    )
    ambiguity: AmbiguityLevel = Field(
        default=AmbiguityLevel.LOW,
        description="Evaluated ambiguity level of the current user message."
    )
    generation_instruction: str = Field(
        ...,
        description="Concrete, high-level instruction guiding the generator."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0.0 to 1.0) for the selected strategy and plan."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return self.model_dump()
