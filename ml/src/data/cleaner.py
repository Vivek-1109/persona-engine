"""
Persona Engine — Personality-Preserving Data Cleaner
Cleans conversation records while explicitly preserving unique personality signals:
emojis, slang, Hinglish, expressive punctuation, capitalization habits, etc.
"""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Optional


class DataCleaner:
    """
    Cleans conversation transcripts without erasing personality traits.
    """

    def __init__(
        self,
        min_turns: int = 2,
        normalize_unicode: bool = True,
        remove_consecutive_duplicates: bool = True,
        preserve_emojis: bool = True,
        preserve_slang: bool = True,
        preserve_language_mixing: bool = True,
    ):
        self.min_turns = min_turns
        self.normalize_unicode = normalize_unicode
        self.remove_consecutive_duplicates = remove_consecutive_duplicates
        self.preserve_emojis = preserve_emojis
        self.preserve_slang = preserve_slang
        self.preserve_language_mixing = preserve_language_mixing

    def clean_text(self, text: str) -> str:
        """
        Cleans a single message text:
        - Strips leading/trailing whitespace
        - Normalizes unicode (NFKC) while preserving characters and symbols
        - Keeps emojis, casing, and expressive punctuation intact
        """
        if not text:
            return ""

        cleaned = text.strip()

        if self.normalize_unicode:
            # NFKC normalizes compatibility characters while keeping emojis and script intact
            cleaned = unicodedata.normalize("NFKC", cleaned)

        return cleaned

    def clean_conversation(self, conv: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Cleans a single conversation dict.
        Returns cleaned conversation or None if conversation fails quality criteria.
        """
        messages = conv.get("messages", [])
        if not messages or not isinstance(messages, list):
            return None

        cleaned_messages: List[Dict[str, Any]] = []
        last_speaker: Optional[str] = None
        last_text: Optional[str] = None

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            raw_speaker = msg.get("speaker")
            raw_text = msg.get("text")

            if not raw_speaker or raw_text is None:
                continue

            cleaned_text = self.clean_text(str(raw_text))
            if not cleaned_text:
                continue

            speaker_str = str(raw_speaker).strip().lower()

            # Optional deduplication of immediately repeated identical messages
            if (
                self.remove_consecutive_duplicates
                and speaker_str == last_speaker
                and cleaned_text == last_text
            ):
                continue

            cleaned_msg = dict(msg)
            cleaned_msg["speaker"] = speaker_str
            cleaned_msg["text"] = cleaned_text

            cleaned_messages.append(cleaned_msg)
            last_speaker = speaker_str
            last_text = cleaned_text

        if len(cleaned_messages) < self.min_turns:
            return None

        result = dict(conv)
        result["messages"] = cleaned_messages
        return result

    def clean_dataset(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cleans a collection of conversations."""
        cleaned_dataset: List[Dict[str, Any]] = []
        for conv in conversations:
            res = self.clean_conversation(conv)
            if res is not None:
                cleaned_dataset.append(res)
        return cleaned_dataset


def clean_conversations(
    conversations: List[Dict[str, Any]],
    min_turns: int = 2,
    normalize_unicode: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience helper to clean a list of conversations."""
    cleaner = DataCleaner(min_turns=min_turns, normalize_unicode=normalize_unicode)
    return cleaner.clean_dataset(conversations)
