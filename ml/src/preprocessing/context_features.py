#!/usr/bin/env python3
"""
Persona Engine — Stage 5: Context Features Extractor
Extracts structural and conversational state metadata from dialogue context windows:
- context_depth (number of messages in the preceding context)
- number_of_turns (count of conversational turns / exchanges)
- speaker_alternation (alternation rate and strict alternation flag)
- conversation_state (opening, inquiry, ongoing, agreement, banter, closing)
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ContextFeatures:
    context_depth: int
    number_of_turns: int
    speaker_alternation_rate: float
    is_strict_alternation: bool
    speaker_sequence: List[str]
    last_user_message: str
    conversation_state: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ContextFeatureExtractor:
    """
    Analyzes conversation structure, speaker dynamics, and conversation state.
    """

    STATE_OPENING_WORDS = {"hi", "hello", "hey", "yo", "sup", "bhai sun", "ek baat sun", "suno", "sun"}
    STATE_CLOSING_WORDS = {"bye", "gn", "good night", "so ja", "kal milte", "nikal rha hu", "so raha hu", "chalo bye"}
    STATE_AGREEMENT_WORDS = {"thik", "theek", "done", "haan", "sahi h", "ok", "cool", "aaja", "chal"}
    STATE_BANTER_WORDS = {"bsdk", "bencho", "saale", "chutiya", "lawda", "pagle", "terese acha", "bot"}

    def extract_features(
        self,
        context: List[Dict[str, str]],
        target_response: str = "",
    ) -> ContextFeatures:
        """
        Extracts structural context features from preceding dialogue context.
        context is expected to be a list of dicts with 'role' and 'content' keys
        (excluding system prompt if desired, or handling system prompt gracefully).
        """
        # Filter out system messages to focus strictly on human/dialogue turns
        dialogue_msgs = [m for m in context if m.get("role") in ("user", "assistant")]
        context_depth = len(dialogue_msgs)

        if not dialogue_msgs:
            return ContextFeatures(
                context_depth=0,
                number_of_turns=0,
                speaker_alternation_rate=1.0,
                is_strict_alternation=True,
                speaker_sequence=[],
                last_user_message="",
                conversation_state="opening",
            )

        # Speaker sequence
        speaker_sequence = [m.get("role", "user") for m in dialogue_msgs]

        # Alternation calculation
        if len(speaker_sequence) <= 1:
            alternation_rate = 1.0
            is_strict = True
        else:
            transitions = sum(1 for i in range(len(speaker_sequence) - 1) if speaker_sequence[i] != speaker_sequence[i + 1])
            max_transitions = len(speaker_sequence) - 1
            alternation_rate = round(transitions / max_transitions, 2)
            is_strict = (alternation_rate == 1.0)

        # Estimate number of full exchanges (turns)
        number_of_turns = max(1, (context_depth + 1) // 2)

        # Identify last user message
        last_user_msg = ""
        for m in reversed(dialogue_msgs):
            if m.get("role") == "user":
                last_user_msg = m.get("content", "").strip()
                break

        # Deterministically infer conversation state
        state = self.infer_conversation_state(
            context_depth=context_depth,
            last_user_msg=last_user_msg,
            target_response=target_response,
            dialogue_msgs=dialogue_msgs,
        )

        return ContextFeatures(
            context_depth=context_depth,
            number_of_turns=number_of_turns,
            speaker_alternation_rate=alternation_rate,
            is_strict_alternation=is_strict,
            speaker_sequence=speaker_sequence,
            last_user_message=last_user_msg,
            conversation_state=state,
        )

    def infer_conversation_state(
        self,
        context_depth: int,
        last_user_msg: str,
        target_response: str,
        dialogue_msgs: List[Dict[str, str]],
    ) -> str:
        """
        Infers state: opening, inquiry, ongoing, agreement, banter, closing.
        """
        last_user_lower = last_user_msg.lower().strip()
        target_lower = target_response.lower().strip()
        combined = f"{last_user_lower} {target_lower}"

        # 1. Closing
        if any(w in target_lower for w in self.STATE_CLOSING_WORDS) or any(w in last_user_lower for w in self.STATE_CLOSING_WORDS):
            return "closing"

        # 2. Banter
        if any(w in combined for w in self.STATE_BANTER_WORDS):
            return "banter"

        # 3. Agreement / Resolution
        if any(w in target_lower for w in self.STATE_AGREEMENT_WORDS) and len(target_lower.split()) <= 4:
            return "agreement"

        # 4. Inquiry (User asked a direct question)
        if "?" in last_user_msg or last_user_lower.startswith(("kya ", "kyu ", "kab ", "kaha ", "kaise ", "why ", "what ")):
            return "inquiry"

        # 5. Opening (Single turn starting conversation)
        if context_depth <= 1 and any(w in last_user_lower for w in self.STATE_OPENING_WORDS):
            return "opening"
        if context_depth <= 1 and len(last_user_msg.split()) <= 2:
            return "opening"

        # 6. Default ongoing multi-turn state
        return "ongoing"
