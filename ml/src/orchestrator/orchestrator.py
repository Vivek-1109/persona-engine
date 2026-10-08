"""
Main Conversation Orchestrator for Persona Engine (Stage 6D).
Acts as the central coordination and decision-making layer between Conversation Input,
Context Engine (Stage 6B), Memory Engine (Stage 6C), and downstream Model Gateway (Stage 6A).
"""

from typing import Any, Dict, List, Optional

from ml.src.context.context_analyzer import ContextAnalyzer
from ml.src.context.context_schema import (
    AmbiguityLevel,
    ContextAnalysis,
    ConversationState,
    TopicCategory,
    UserIntent,
)
from ml.src.memory.memory_retriever import MemoryRetriever
from ml.src.orchestrator.orchestrator_schema import (
    ConversationRequest,
    ResponsePlan,
    ResponseStrategy,
    ResponseTone,
)
from ml.src.orchestrator.response_planner import ResponsePlanner


class ConversationOrchestrator:
    """
    Coordinates context intelligence, persistent memory retrieval,
    dialogue strategy selection, and response planning.
    """

    def __init__(
        self,
        context_analyzer: Optional[ContextAnalyzer] = None,
        memory_retriever: Optional[MemoryRetriever] = None,
    ):
        self.context_analyzer = context_analyzer or ContextAnalyzer()
        self.memory_retriever = memory_retriever

    def _extract_last_user_message(self, messages: List[Dict[str, str]]) -> str:
        """Finds the most recent user turn content."""
        for msg in reversed(messages):
            if msg.get("role", "").lower() in {"user", "human"}:
                return msg.get("content", "").strip()
        return ""

    def plan(self, request: ConversationRequest) -> ResponsePlan:
        """
        Processes a ConversationRequest and produces a structured ResponsePlan.
        Safely handles missing or partial data with deterministic fallback policies.
        """
        messages = request.messages or []
        last_user_msg = self._extract_last_user_message(messages)

        # 1. Fallback for completely empty messages
        if not messages and not request.context:
            return ResponsePlan(
                topic=TopicCategory.CASUAL_CHAT,
                user_intent=UserIntent.UNKNOWN,
                conversation_state=ConversationState.OPENING,
                response_strategy=ResponseStrategy.REACT,
                tone=ResponseTone.CASUAL,
                use_memory=False,
                selected_memory_ids=[],
                selected_memories=[],
                context_depth=0,
                ambiguity=AmbiguityLevel.HIGH,
                generation_instruction="Greet the user and initiate a friendly, casual dialogue.",
                confidence=0.30,
            )

        # 2. Resolve ContextAnalysis
        context = request.context
        if context is None:
            if messages:
                context = self.context_analyzer.analyze(messages)
            else:
                context = ContextAnalysis(
                    topic=TopicCategory.CASUAL_CHAT,
                    conversation_state=ConversationState.ONGOING,
                    user_intent=UserIntent.CASUAL_CHAT,
                    ambiguity=AmbiguityLevel.LOW,
                    recent_activity=None,
                    context_depth=len(messages),
                    last_user_message=last_user_msg,
                    speaker_alternation_rate=1.0,
                )

        # 3. Resolve Candidate Memories
        candidate_memories = request.memories
        if candidate_memories is None and self.memory_retriever is not None:
            candidate_memories = self.memory_retriever.retrieve(
                persona_id=request.persona_id,
                conversation_context=messages,
                context_analysis=context,
            )

        # 4. Construct Response Plan
        return ResponsePlanner.build_plan(
            topic=context.topic,
            user_intent=context.user_intent,
            conversation_state=context.conversation_state,
            ambiguity=context.ambiguity,
            last_user_message=context.last_user_message or last_user_msg,
            context_depth=context.context_depth,
            recent_activity=context.recent_activity,
            candidate_memories=candidate_memories,
        )

    def orchestrate(self, request: ConversationRequest) -> ResponsePlan:
        """Alias for plan() to maintain unified orchestrator interface."""
        return self.plan(request)
