"""
Deterministic Rule-Based Memory Extractor for Persona Engine (Stage 6C).
Extracts durable facts, preferences, goals, plans, relationships, experiences, and events
from conversational utterances. Rejects ephemeral turns and speculative inferences.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from ml.src.memory.memory_schema import (
    ImportanceLevel,
    Memory,
    MemoryType,
)


class MemoryExtractor:
    """Deterministic, explainable extractor for persistent memory items."""

    # Ephemeral turns that MUST NOT generate persistent memories
    NON_MEMORY_PATTERNS = [
        r"^(haan|haa|haan bhai|haa bhai|theek|thik|thik hai|ok|okh|done|yep|yes|no|nhi|nahi|naa)$",
        r"^(aaja|aao|chal|chalo|aaya|suna|arre yaar|kya|kaha|kyu|kaise)$",
        r"^aaja\s+(game\s+)?(khelte\s+hain|khele|chal)$",
        r"^(khelega\??|khelenge\??|chalega\??|game\s+aaja)$",
        r"^(kal\s+milte\s+hain|baad\s+me\s+milte\s+hain|canteen\s+chale\??)$",
        r"^(kya\s+scene\s+hai|kya\s+chal\s+raha\s+hai|bore\s+ho\s+raha\s+hu)$",
    ]

    # Extraction patterns: (Regex, MemoryType, Content Template, Topic, Importance)
    EXTRACTION_RULES: List[Tuple[str, MemoryType, str, str, ImportanceLevel]] = [
        # 1. Facts: Education & Identity
        (
            r"\b(main|i(\s+am)?)\s+b\.?tech(\s+cse)?\s+(kar\s+raha\s+hu|study|doing)\b",
            MemoryType.FACT, "Studies B.Tech CSE", "college", ImportanceLevel.HIGH
        ),
        (
            r"\b(my\s+college|mera\s+college)\s+(is\s+in\s+noida|noida\s+(me\s+)?hai|hai\s+noida)\b",
            MemoryType.FACT, "College is located in Noida", "college", ImportanceLevel.MEDIUM
        ),
        (
            r"\b(main|i\s+live\s+in)\s+delhi\s+(me\s+rehta\s+hu|resident)?\b",
            MemoryType.FACT, "Lives in Delhi", "casual_chat", ImportanceLevel.MEDIUM
        ),
        (
            r"\b(acer\s+nitro|laptop)\s+(le\s+liya|bought|purchased|have)\b",
            MemoryType.FACT, "Owns an Acer Nitro laptop", "technology", ImportanceLevel.MEDIUM
        ),

        # 2. Preferences
        (
            r"\b(mujhe|i(\s+really)?)\s+(love|like|pasand)\s+(hai\s+|hain\s+)?gaming\b|"
            r"\b(gaming|games)\s+(bahut\s+)?pasand\s+hai\b",
            MemoryType.PREFERENCE, "Likes gaming", "gaming", ImportanceLevel.HIGH
        ),
        (
            r"\b(mujhe|i\s+prefer|i\s+like)\s+java\s+(bahut\s+)?pasand\s+hai\b|"
            r"\b(i\s+prefer\s+java|prefer\s+java)\b",
            MemoryType.PREFERENCE, "Prefers Java", "technology", ImportanceLevel.HIGH
        ),
        (
            r"\b(mujhe|i\s+prefer|i\s+like)\s+python\s+(bahut\s+)?pasand\s+hai\b|"
            r"\b(i\s+prefer\s+python|prefer\s+python)\b",
            MemoryType.PREFERENCE, "Prefers Python", "technology", ImportanceLevel.HIGH
        ),
        (
            r"\b(marvel\s+movies|marvel)\s+(bahut\s+)?pasand\s+hai\b|"
            r"\b(i\s+love\s+marvel\s+movies|like\s+marvel\s+movies)\b|"
            r"\bmujhe\s+marvel\s+movies\s+(bahut\s+)?pasand\s+hai\b",
            MemoryType.PREFERENCE, "Likes Marvel movies", "movies", ImportanceLevel.HIGH
        ),
        (
            r"\b(bgmi|valorant)\s+(bahut\s+)?pasand\s+hai\b|"
            r"\b(i\s+love\s+bgmi|like\s+bgmi)\b",
            MemoryType.PREFERENCE, "Likes BGMI", "gaming", ImportanceLevel.MEDIUM
        ),

        # 3. Goals
        (
            r"\b(backend\s+developer|java\s+developer)\s+(banna\s+chahta\s+hu|banna\s+hai)\b|"
            r"\b(want\s+to\s+become\s+a\s+(backend|java)\s+developer)\b",
            MemoryType.GOAL, "Wants to become a backend developer", "technology", ImportanceLevel.HIGH
        ),
        (
            r"\b(preparing\s+for\s+interviews|interviews?\s+ki\s+taiyari\s+kar\s+raha\s+hu)\b",
            MemoryType.GOAL, "Preparing for job interviews", "college", ImportanceLevel.MEDIUM
        ),
        (
            r"\b(spring\s+boot\s+seekh\s+raha\s+hu|learning\s+spring\s+boot)\b",
            MemoryType.GOAL, "Learning Spring Boot", "technology", ImportanceLevel.MEDIUM
        ),

        # 4. Plans
        (
            r"\b(kal|tomorrow|next\s+monday)\s+(interview\s+hai|have\s+an\s+interview)\b|"
            r"\b(have\s+an\s+interview\s+(tomorrow|next\s+monday))\b",
            MemoryType.PLAN, "Has an upcoming interview", "college", ImportanceLevel.LOW
        ),
        (
            r"\b(next\s+week\s+(i\'ll\s+visit|jaunga)\s+delhi|delhi\s+visit\s+karega)\b",
            MemoryType.PLAN, "Planning to visit Delhi next week", "plans", ImportanceLevel.LOW
        ),

        # 5. Relationships
        (
            r"\b(rahul\s+(is\s+my|mera)\s+college\s+friend\s+(hai)?)\b",
            MemoryType.RELATIONSHIP, "Rahul is a college friend", "college", ImportanceLevel.HIGH
        ),
        (
            r"\b(she\s+is\s+my\s+sister|meri\s+behen\s+hai)\b",
            MemoryType.RELATIONSHIP, "Has a sister", "social", ImportanceLevel.HIGH
        ),
        (
            r"\b(amit\s+(is\s+my|mera)\s+roommate\s+(hai)?)\b",
            MemoryType.RELATIONSHIP, "Amit is a roommate", "college", ImportanceLevel.MEDIUM
        ),

        # 6. Experiences
        (
            r"\b(participated\s+in\s+sih|sih\s+me\s+participate\s+kiya\s+tha)\b",
            MemoryType.EXPERIENCE, "Participated in Smart India Hackathon (SIH)", "college", ImportanceLevel.MEDIUM
        ),
        (
            r"\b(hackathon\s+project\s+(pe\s+kaam\s+kiya|worked\s+on))\b",
            MemoryType.EXPERIENCE, "Worked on a hackathon project", "technology", ImportanceLevel.MEDIUM
        ),
        (
            r"\b(gaming\s+tournament\s+(me\s+participate|khela\s+tha)|participated\s+in\s+gaming\s+tournament)\b",
            MemoryType.EXPERIENCE, "Participated in a gaming tournament", "gaming", ImportanceLevel.MEDIUM
        ),

        # 7. Events
        (
            r"\b(internship\s+started|internship\s+shuru\s+ho\s+gayi)\b",
            MemoryType.EVENT, "Internship started", "college", ImportanceLevel.MEDIUM
        ),
    ]

    @classmethod
    def is_ephemeral_or_non_memory(cls, text: str) -> bool:
        """Checks whether the utterance is purely ephemeral conversation."""
        clean = text.strip().lower()
        if not clean:
            return True

        for pat in cls.NON_MEMORY_PATTERNS:
            if re.match(pat, clean):
                return True

        return False

    @classmethod
    def extract_from_text(
        cls,
        text: str,
        persona_id: str = "vivek",
        source_conversation_id: Optional[str] = None,
        source_message_id: Optional[Union[str, int]] = None,
    ) -> List[Memory]:
        """
        Extracts candidate memories from a single text utterance.
        """
        clean = text.strip().lower()

        # Ephemeral check
        if cls.is_ephemeral_or_non_memory(clean):
            return []

        candidates: List[Memory] = []

        for pat, m_type, content, topic, importance in cls.EXTRACTION_RULES:
            if re.search(pat, clean):
                mem = Memory(
                    persona_id=persona_id,
                    memory_type=m_type,
                    content=content,
                    topic=topic,
                    importance=importance,
                    source_conversation_id=source_conversation_id,
                    source_message_id=source_message_id,
                )
                candidates.append(mem)

        return candidates

    @classmethod
    def extract_from_conversation(
        cls,
        messages: List[Dict[str, str]],
        persona_id: str = "vivek",
        source_conversation_id: Optional[str] = None,
    ) -> List[Memory]:
        """
        Extracts candidate memories from multi-turn dialogue context.
        Inspects user utterances with priority.
        """
        all_extracted: List[Memory] = []
        seen_contents = set()

        for idx, m in enumerate(messages):
            role = m.get("role", "").lower()
            content = m.get("content", "").strip()

            # Focus memory extraction on user statements
            if role in {"user", "human"}:
                extracted = cls.extract_from_text(
                    text=content,
                    persona_id=persona_id,
                    source_conversation_id=source_conversation_id,
                    source_message_id=idx,
                )
                for mem in extracted:
                    if mem.content not in seen_contents:
                        seen_contents.add(mem.content)
                        all_extracted.append(mem)

        return all_extracted
