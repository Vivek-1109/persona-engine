"""
Response planner for Persona Engine Conversation Orchestrator (Stage 6D).
Synthesizes strategy, tone, memory policy, ambiguity handling, confidence scoring,
and generation instructions into a strongly typed ResponsePlan.
"""

import re
from typing import Any, Dict, List, Optional

from ml.src.context.context_schema import (
    AmbiguityLevel,
    ConversationState,
    TopicCategory,
    UserIntent,
)
from ml.src.orchestrator.memory_policy import MemoryPolicy
from ml.src.orchestrator.orchestrator_schema import (
    ResponsePlan,
    ResponseStrategy,
    ResponseTone,
)
from ml.src.orchestrator.strategy_selector import StrategySelector


class ResponsePlanner:
    """
    Constructs a complete ResponsePlan from conversational dimensions.
    Deterministic, explainable, and free of external LLM dependencies.
    """

    @classmethod
    def select_tone(
        cls,
        topic: TopicCategory,
        conversation_state: ConversationState,
        user_intent: UserIntent,
        last_user_message: str,
    ) -> ResponseTone:
        """
        Determines the appropriate high-level communicative tone.
        """
        clean_msg = last_user_message.lower()

        # 1. Banter state -> Humorous or Teasing
        if conversation_state == ConversationState.BANTER:
            if user_intent == UserIntent.DISAGREEMENT or re.search(r"\b(arre|abe|chal\s+na)\b", clean_msg):
                return ResponseTone.TEASING
            return ResponseTone.HUMOROUS

        # 2. College topic with formal notice or critical updates -> Serious or Neutral
        if topic == TopicCategory.COLLEGE:
            if re.search(r"\b(notice|attendance|dean|exam|strict|detain|warning)\b", clean_msg):
                return ResponseTone.SERIOUS
            return ResponseTone.NEUTRAL

        # 3. Technology with problem/defect -> Supportive or Casual
        if topic == TopicCategory.TECHNOLOGY:
            if re.search(r"\b(drain|battery|problem|issue|broken|crash|bug|slow)\b", clean_msg):
                return ResponseTone.SUPPORTIVE
            return ResponseTone.CASUAL

        # 4. Closing or standard social turns -> Casual
        if conversation_state == ConversationState.CLOSING:
            return ResponseTone.CASUAL

        # 5. Gaming and Movies -> Casual
        if topic in {TopicCategory.GAMING, TopicCategory.MOVIES}:
            return ResponseTone.CASUAL

        return ResponseTone.CASUAL

    @classmethod
    def calculate_confidence(
        cls,
        user_intent: UserIntent,
        ambiguity: AmbiguityLevel,
        context_depth: int,
        response_strategy: ResponseStrategy,
        use_memory: bool,
    ) -> float:
        """
        Computes an explainable, deterministic confidence rating in [0.10, 1.00].
        Formula: Intent Clarity + Ambiguity Resolution + Depth + Strategy Certainty + Memory Support
        """
        # 1. Intent clarity (0.15 - 0.25)
        if user_intent in {
            UserIntent.QUESTION,
            UserIntent.INVITATION,
            UserIntent.AGREEMENT,
            UserIntent.PLANNING,
            UserIntent.DISAGREEMENT,
        }:
            c_intent = 0.25
        elif user_intent in {UserIntent.INFORMATION, UserIntent.REACTION}:
            c_intent = 0.20
        else:
            c_intent = 0.15

        # 2. Ambiguity resolution (0.05 - 0.35)
        if ambiguity == AmbiguityLevel.LOW:
            c_ambiguity = 0.35
        elif ambiguity == AmbiguityLevel.MEDIUM:
            c_ambiguity = 0.20
        else:
            c_ambiguity = 0.05

        # 3. Context depth (0.05 - 0.15)
        if context_depth >= 3:
            c_depth = 0.15
        elif context_depth == 2:
            c_depth = 0.10
        else:
            c_depth = 0.05

        # 4. Strategy certainty (0.10 - 0.15)
        if response_strategy in {
            ResponseStrategy.ACCEPT,
            ResponseStrategy.DECLINE,
            ResponseStrategy.CLOSE_CONVERSATION,
            ResponseStrategy.ANSWER,
            ResponseStrategy.PROVIDE_INFORMATION,
        }:
            c_strategy = 0.15
        else:
            c_strategy = 0.10

        # 5. Memory support (0.05 - 0.10)
        c_memory = 0.10 if use_memory else 0.05

        raw_confidence = c_intent + c_ambiguity + c_depth + c_strategy + c_memory
        return round(max(0.10, min(1.00, raw_confidence)), 2)

    @classmethod
    def generate_instruction(
        cls,
        response_strategy: ResponseStrategy,
        tone: ResponseTone,
        topic: TopicCategory,
        user_intent: UserIntent,
        use_memory: bool,
    ) -> str:
        """
        Creates concise, structured guidance for downstream Stage 6A text generation.
        """
        memory_clause = (
            " Incorporate relevant context naturally without explicitly reciting memory records."
            if use_memory
            else " Rely on current dialogue context."
        )

        strategy_instructions = {
            ResponseStrategy.ACCEPT: f"Respond with a {tone.value} acceptance to the {topic.value} invitation. Keep it natural and concise.{memory_clause}",
            ResponseStrategy.DECLINE: f"Respond with a {tone.value} decline while keeping rapport friendly. State the reason briefly.{memory_clause}",
            ResponseStrategy.ANSWER: f"Provide a clear, {tone.value} answer addressing the {topic.value} question or discussion.{memory_clause}",
            ResponseStrategy.ACKNOWLEDGE: f"Acknowledge the user's remark with a {tone.value} response to keep conversational flow smooth.{memory_clause}",
            ResponseStrategy.ASK_CLARIFICATION: f"Inquire for clarification in a {tone.value} manner. The user statement is ambiguous; politely ask what they mean.",
            ResponseStrategy.SUGGEST: f"Offer a helpful, {tone.value} suggestion or practical advice regarding {topic.value}.{memory_clause}",
            ResponseStrategy.REACT: f"Express an authentic, {tone.value} reaction matching the conversational mood.{memory_clause}",
            ResponseStrategy.CONTINUE_BANTER: f"Continue the playful banter with a {tone.value}, witty retort.{memory_clause}",
            ResponseStrategy.PROVIDE_INFORMATION: f"Share relevant, {tone.value} information regarding the {topic.value} update.{memory_clause}",
            ResponseStrategy.CLOSE_CONVERSATION: f"Acknowledge the wrap-up and conclude the dialogue naturally with a brief {tone.value} sign-off.",
        }

        return strategy_instructions.get(
            response_strategy,
            f"Generate an authentic {tone.value} response adhering to the persona profile.{memory_clause}",
        )

    @classmethod
    def build_plan(
        cls,
        topic: TopicCategory,
        user_intent: UserIntent,
        conversation_state: ConversationState,
        ambiguity: AmbiguityLevel,
        last_user_message: str,
        context_depth: int,
        recent_activity: Optional[str] = None,
        candidate_memories: Optional[List[Any]] = None,
    ) -> ResponsePlan:
        """
        Assembles all components into a validated ResponsePlan.
        """
        # 1. Evaluate memory candidates
        use_memory, selected_memory_ids, selected_memories = MemoryPolicy.evaluate_memories(
            candidate_memories=candidate_memories,
            current_topic=topic,
            conversation_state=conversation_state,
        )

        # 2. Select strategy
        strategy = StrategySelector.select_strategy(
            user_intent=user_intent,
            conversation_state=conversation_state,
            ambiguity=ambiguity,
            topic=topic,
            last_user_message=last_user_message,
            recent_activity=recent_activity,
            has_relevant_memories=use_memory,
        )

        # Suppress memories if wrapping up conversation
        if strategy == ResponseStrategy.CLOSE_CONVERSATION:
            use_memory = False
            selected_memory_ids = []
            selected_memories = []

        # 3. Select tone
        tone = cls.select_tone(
            topic=topic,
            conversation_state=conversation_state,
            user_intent=user_intent,
            last_user_message=last_user_message,
        )

        # 4. Calculate confidence
        confidence = cls.calculate_confidence(
            user_intent=user_intent,
            ambiguity=ambiguity,
            context_depth=context_depth,
            response_strategy=strategy,
            use_memory=use_memory,
        )

        # 5. Create generation instruction
        generation_instruction = cls.generate_instruction(
            response_strategy=strategy,
            tone=tone,
            topic=topic,
            user_intent=user_intent,
            use_memory=use_memory,
        )

        return ResponsePlan(
            topic=topic,
            user_intent=user_intent,
            conversation_state=conversation_state,
            response_strategy=strategy,
            tone=tone,
            use_memory=use_memory,
            selected_memory_ids=selected_memory_ids,
            selected_memories=selected_memories,
            context_depth=context_depth,
            ambiguity=ambiguity,
            generation_instruction=generation_instruction,
            confidence=confidence,
        )
