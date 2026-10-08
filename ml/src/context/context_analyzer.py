"""
Context Analyzer coordinating module for Persona Engine Context Engine.
Produces the comprehensive ContextAnalysis structure.
Deterministic, fast, testable, and framework-free.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from ml.src.context.context_schema import (
    AmbiguityLevel,
    ContextAnalysis,
    ConversationState,
    TopicCategory,
    UserIntent,
)
from ml.src.context.conversation_state import ConversationStateClassifier
from ml.src.context.intent_analyzer import IntentAnalyzer
from ml.src.context.topic_analyzer import TopicAnalyzer


class ContextAnalyzer:
    """Central engine performing runtime contextual analysis on conversation windows."""

    # Words and phrases that exhibit high context dependence when standing alone
    HIGH_AMBIGUITY_PATTERNS = [
        r"^aaja$", r"^khelega\??$", r"^game aaja$", r"^thik$", r"^theek$",
        r"^sahi h$", r"^sahi hai$", r"^kya\??$", r"^kaha\??$", r"^aaya$",
        r"^nhi bhai$", r"^bsdk$", r"^ok$", r"^okh$", r"^done$", r"^chal$",
        r"^arre yaar$", r"^suna$"
    ]

    @classmethod
    def compute_ambiguity(cls, last_user_message: str) -> AmbiguityLevel:
        """
        Calculates context dependence of the current user message.
        Terse, unanchored utterances receive 'high'; explicit full queries receive 'low'.
        """
        clean = last_user_message.strip().lower()
        words = clean.split()

        # Direct pattern match for known ambiguous anchors
        for pat in cls.HIGH_AMBIGUITY_PATTERNS:
            if re.match(pat, clean):
                return AmbiguityLevel.HIGH

        # Utterances of 1 or 2 words without specific noun entities
        if len(words) <= 2 and not any(w in clean for w in ["laptop", "college", "attendance", "processor", "fifa", "bgmi"]):
            return AmbiguityLevel.HIGH

        # Questions or invitations with some specificity
        if len(words) <= 5 and any(w in clean for w in ["khelega", "aayega", "milte", "chale", "scene"]):
            return AmbiguityLevel.MEDIUM

        # Explicit informative inquiries
        return AmbiguityLevel.LOW

    @classmethod
    def extract_recent_activity(cls, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Extracts a concise, factual description of immediately preceding conversational activity.
        Returns None if no specific activity is detected. Does not hallucinate.
        """
        if len(messages) < 2:
            return None

        # Inspect preceding turns (up to 3 prior turns)
        preceding_text = " ".join(m.get("content", "").lower() for m in messages[:-1])

        if re.search(r"\bassignment\b", preceding_text):
            return "completing an assignment"
        if re.search(r"\bcanteen\b", preceding_text):
            return "meeting at canteen"
        if re.search(r"\bdinner\b|\bkhana\b", preceding_text):
            return "dinner / meal"
        if re.search(r"\bclass\b|\blecture\b", preceding_text):
            return "attending class / lecture"
        if re.search(r"\blibrary\b", preceding_text):
            return "at the library"
        if re.search(r"\bexam\b|\badmit card\b", preceding_text):
            return "exam preparation"
        if re.search(r"\bgaming\b|\bbgmi\b|\bvalorant\b", preceding_text):
            return "gaming session"
        if re.search(r"\bcharg(ing|er)\b|\bbattery\b", preceding_text):
            return "phone charging"
        if re.search(r"\b(laptop|nitro|acer)\b", preceding_text):
            return "discussing laptop purchase"

        return None

    @classmethod
    def compute_speaker_alternation_rate(cls, messages: List[Dict[str, str]]) -> float:
        """
        Calculates speaker alternation frequency:
        switches / (total_messages - 1).
        """
        if len(messages) <= 1:
            return 1.0

        roles = [m.get("role", "").strip().lower() for m in messages if m.get("content", "").strip()]
        if len(roles) <= 1:
            return 1.0

        switches = sum(1 for i in range(1, len(roles)) if roles[i] != roles[i - 1])
        transitions = len(roles) - 1
        return round(switches / transitions, 2)

    @classmethod
    def analyze(cls, messages: List[Dict[str, str]]) -> ContextAnalysis:
        """
        Main entry point. Analyzes recent conversation window and returns
        the strongly-typed ContextAnalysis report.
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list of turn objects")

        # Extract last user message
        last_user_message = ""
        for m in reversed(messages):
            if m.get("role", "").strip().lower() == "user":
                last_user_message = m.get("content", "").strip()
                break

        if not last_user_message:
            last_user_message = messages[-1].get("content", "").strip()

        # 1. Topic analysis across full context
        topic = TopicAnalyzer.analyze(messages)

        # 2. Conversation state classification
        conv_state = ConversationStateClassifier.classify(messages)

        # 3. User intent classification
        intent = IntentAnalyzer.analyze(messages)

        # 4. Context ambiguity calculation
        ambiguity = cls.compute_ambiguity(last_user_message)

        # 5. Recent activity extraction
        recent_activity = cls.extract_recent_activity(messages)

        # 6. Context depth
        context_depth = len(messages)

        # 7. Speaker alternation rate
        alt_rate = cls.compute_speaker_alternation_rate(messages)

        return ContextAnalysis(
            topic=topic,
            conversation_state=conv_state,
            user_intent=intent,
            ambiguity=ambiguity,
            recent_activity=recent_activity,
            context_depth=context_depth,
            last_user_message=last_user_message,
            speaker_alternation_rate=alt_rate
        )
