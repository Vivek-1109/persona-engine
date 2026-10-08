"""
Conversation State Classifier for Persona Engine Context Engine.
Infers dialogue progression state based on multi-turn structure and semantic acts.
"""

import re
from typing import Any, Dict, List

from ml.src.context.context_schema import ConversationState


class ConversationStateClassifier:
    """Classifies conversation state using structural context and dialogue acts."""

    BANTER_MARKERS = [
        r"\b(bsdk|bencho|bc|chutiya|saale|bot|noob)\b",
        r"😂|🤣|💀|😆"
    ]

    CLOSING_MARKERS = [
        r"\b(bye|gn|good night|chalta hu|so raha|baad me milte|alvida)\b"
    ]

    AGREEMENT_MARKERS = [
        r"^(thik|theek|haan|haa|sahi h|sahi hai|ok|okh|done|chal)\b"
    ]

    @classmethod
    def classify(cls, messages: List[Dict[str, str]]) -> ConversationState:
        """
        Classifies the ongoing dialogue state over the conversation window.
        """
        if not messages:
            return ConversationState.OPENING

        # Single message without prior context is an opening
        if len(messages) == 1:
            return ConversationState.OPENING

        last_turn = messages[-1].get("content", "").strip().lower()
        all_text = " ".join(m.get("content", "").lower() for m in messages)

        # 1. Closing check
        for pat in cls.CLOSING_MARKERS:
            if re.search(pat, last_turn):
                return ConversationState.CLOSING

        # 2. Banter check across recent context
        banter_count = sum(1 for pat in cls.BANTER_MARKERS if re.search(pat, all_text))
        if banter_count >= 1 and any(re.search(pat, last_turn) for pat in cls.BANTER_MARKERS):
            return ConversationState.BANTER

        # 3. Agreement check (short affirmation following prior turn)
        for pat in cls.AGREEMENT_MARKERS:
            if re.search(pat, last_turn):
                return ConversationState.AGREEMENT

        # 4. Inquiry check (active questions in the current turn)
        if "?" in last_turn or any(q in last_turn for q in ["kya", "kyu", "kaha", "kab", "kaise", "kitna"]):
            return ConversationState.INQUIRY

        # 5. Default ongoing conversation
        return ConversationState.ONGOING
