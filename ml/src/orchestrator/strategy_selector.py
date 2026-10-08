"""
Deterministic strategy selector for Persona Engine Conversation Orchestrator (Stage 6D).
Selects the appropriate ResponseStrategy based on intent, conversational state,
ambiguity level, topic, and recent dialogue dynamics without calling LLMs.
"""

import re
from typing import Optional

from ml.src.context.context_schema import (
    AmbiguityLevel,
    ConversationState,
    TopicCategory,
    UserIntent,
)
from ml.src.orchestrator.orchestrator_schema import ResponseStrategy


class StrategySelector:
    """
    Evaluates conversational signals and selects a deterministic dialogue action strategy.
    Fully explainable and rule-based.
    """

    @classmethod
    def select_strategy(
        cls,
        user_intent: UserIntent,
        conversation_state: ConversationState,
        ambiguity: AmbiguityLevel,
        topic: TopicCategory,
        last_user_message: str,
        recent_activity: Optional[str] = None,
        has_relevant_memories: bool = False,
    ) -> ResponseStrategy:
        """
        Determines the optimal ResponseStrategy.

        Priority order:
        1. Closing signals -> CLOSE_CONVERSATION
        2. High ambiguity resolution (clarification vs contextual acceptance)
        3. Intent-driven strategies (question -> answer, invitation -> accept/decline, etc.)
        4. Topic-informed fallbacks
        """
        clean_msg = last_user_message.strip().lower()

        # 1. Closing State
        if conversation_state == ConversationState.CLOSING or re.search(
            r"\b(bye|baad\s+me\s+baat|see\s+you|chalta\s+hu|chal\s+baad\s+me|goodnight|tata)\b",
            clean_msg,
        ):
            return ResponseStrategy.CLOSE_CONVERSATION

        # 2. Specific Quick Responses
        if re.match(r"^(thik|theek|ok|okh|done|sahi\s+hai|theek\s+hai)$", clean_msg):
            return ResponseStrategy.ACKNOWLEDGE

        if re.match(r"^(nhi\s+bhai|nahi\s+bhai|naa\s+bhai|no\s+bro)$", clean_msg):
            # If the user is declining or reacting negatively
            if user_intent == UserIntent.INVITATION:
                return ResponseStrategy.DECLINE
            return ResponseStrategy.REACT

        # 3. High Ambiguity Handling
        if ambiguity == AmbiguityLevel.HIGH:
            # Check if context can resolve the ambiguity
            if topic == TopicCategory.GAMING and (
                user_intent == UserIntent.INVITATION
                or re.search(r"\b(aaja|khelega|khele)\b", clean_msg)
            ):
                # Context resolves "Aaja" or "Khelega" in gaming domain
                return ResponseStrategy.ACCEPT

            if user_intent == UserIntent.INVITATION and recent_activity and "assignment" in recent_activity.lower():
                # Conflict in context: friend is busy with assignment
                return ResponseStrategy.ACKNOWLEDGE

            # Unresolvable ambiguity without sufficient domain context
            if topic in {TopicCategory.CASUAL_CHAT, TopicCategory.OTHER} and user_intent in {
                UserIntent.UNKNOWN,
                UserIntent.INVITATION,
            }:
                if re.match(r"^(aaja|chal|kaha|kyu)$", clean_msg):
                    return ResponseStrategy.ASK_CLARIFICATION

        # 4. Questions always get answered (unless ambiguous)
        if user_intent == UserIntent.QUESTION:
            if ambiguity == AmbiguityLevel.HIGH and not clean_msg:
                return ResponseStrategy.ASK_CLARIFICATION
            return ResponseStrategy.ANSWER

        # 5. Domain Problem / Notice Topic Triggers (for statements or reports)
        if topic == TopicCategory.TECHNOLOGY and re.search(
            r"\b(drain|battery|problem|issue|bug|slow|crash)\b", clean_msg
        ):
            return ResponseStrategy.SUGGEST

        if topic == TopicCategory.COLLEGE and re.search(
            r"\b(notice|attendance|dean|exam|strict|detain|warning)\b", clean_msg
        ):
            return ResponseStrategy.PROVIDE_INFORMATION

        # 6. Intent-Driven Strategy Selection
        if user_intent == UserIntent.INVITATION:
            # Standard invitation response
            return ResponseStrategy.ACCEPT

        if user_intent == UserIntent.AGREEMENT:
            if conversation_state == ConversationState.BANTER:
                return ResponseStrategy.CONTINUE_BANTER
            return ResponseStrategy.ACKNOWLEDGE

        if user_intent == UserIntent.DISAGREEMENT:
            if conversation_state == ConversationState.BANTER:
                return ResponseStrategy.CONTINUE_BANTER
            if topic in {TopicCategory.COLLEGE, TopicCategory.TECHNOLOGY}:
                return ResponseStrategy.ANSWER
            return ResponseStrategy.REACT

        if user_intent == UserIntent.PLANNING:
            return ResponseStrategy.SUGGEST

        if user_intent == UserIntent.INFORMATION:
            if topic == TopicCategory.COLLEGE:
                return ResponseStrategy.PROVIDE_INFORMATION
            if topic == TopicCategory.TECHNOLOGY:
                return ResponseStrategy.SUGGEST
            if topic in {TopicCategory.MOVIES, TopicCategory.CASUAL_CHAT}:
                return ResponseStrategy.REACT
            return ResponseStrategy.PROVIDE_INFORMATION

        if user_intent == UserIntent.REACTION:
            if conversation_state == ConversationState.BANTER:
                return ResponseStrategy.CONTINUE_BANTER
            return ResponseStrategy.REACT

        if user_intent == UserIntent.CASUAL_CHAT:
            if conversation_state == ConversationState.BANTER:
                return ResponseStrategy.CONTINUE_BANTER
            if conversation_state == ConversationState.OPENING:
                return ResponseStrategy.ACKNOWLEDGE
            return ResponseStrategy.REACT

        # 5. Default Fallbacks by Conversation State
        if conversation_state == ConversationState.INQUIRY:
            return ResponseStrategy.ANSWER
        if conversation_state == ConversationState.BANTER:
            return ResponseStrategy.CONTINUE_BANTER
        if conversation_state == ConversationState.AGREEMENT:
            return ResponseStrategy.ACKNOWLEDGE

        return ResponseStrategy.REACT
