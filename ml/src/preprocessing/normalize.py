"""
Persona Engine — Personality-Preserving Text Normalization
Normalizes unicode, spacing, and speaker identities while explicitly preserving
stylistic signatures such as capitalization style, emojis, punctuation emphasis,
slang, and Hinglish vocabulary.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional


class TextNormalizer:
    """
    Normalizes conversational text while protecting stylistic identity.
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        collapse_spaces: bool = True,
        preserve_linebreaks: bool = True,
    ):
        self.normalize_unicode = normalize_unicode
        self.collapse_spaces = collapse_spaces
        self.preserve_linebreaks = preserve_linebreaks

    def normalize(self, text: str) -> str:
        """
        Normalizes text without stripping emojis or stylistic patterns.
        """
        if not text:
            return ""

        result = text

        if self.normalize_unicode:
            result = unicodedata.normalize("NFKC", result)

        if self.collapse_spaces:
            if self.preserve_linebreaks:
                # Collapse repeated horizontal whitespace but keep line breaks
                lines = result.splitlines()
                cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
                result = "\n".join(cleaned_lines).strip()
            else:
                result = re.sub(r"\s+", " ", result).strip()
        else:
            result = result.strip()

        return result


def normalize_text(text: str) -> str:
    """Convenience helper to normalize a message text."""
    normalizer = TextNormalizer()
    return normalizer.normalize(text)


def normalize_speaker(
    raw_speaker: str,
    speaker_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Maps speaker names/labels to standard vocabulary: 'user', 'persona', 'system'.
    """
    if not raw_speaker:
        return "user"

    cleaned = str(raw_speaker).strip().lower()

    default_map = {
        "user": "user",
        "human": "user",
        "friend": "user",
        "them": "user",
        "other": "user",
        "persona": "persona",
        "me": "persona",
        "bot": "persona",
        "assistant": "persona",
        "target": "persona",
        "system": "system",
    }

    if speaker_map:
        merged_map = {**default_map, **{k.lower(): v.lower() for k, v in speaker_map.items()}}
    else:
        merged_map = default_map

    return merged_map.get(cleaned, cleaned)
