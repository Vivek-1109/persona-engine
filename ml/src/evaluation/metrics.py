"""
Persona Engine — Evaluation Metrics
Calculates style similarity, vocabulary overlap, emoji consistency,
and response length alignment between generated outputs and reference persona messages.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

EMOJI_REGEX = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50\u2b55\u200d\ufe0f]",
    flags=re.UNICODE,
)


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def calculate_length_similarity(gen_text: str, ref_text: str) -> float:
    """
    Measures how closely generated response length matches reference length.
    Returns float in range [0.0, 1.0], where 1.0 is exact match.
    Formula: min(len1, len2) / max(len1, len2)
    """
    l1 = len(gen_text.strip())
    l2 = len(ref_text.strip())
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return round(min(l1, l2) / max(l1, l2), 4)


def calculate_vocabulary_overlap(gen_text: str, ref_text: str) -> float:
    """
    Computes Jaccard word vocabulary similarity between generated and reference text.
    Returns float in range [0.0, 1.0].
    """
    tokens_gen = _tokenize(gen_text)
    tokens_ref = _tokenize(ref_text)
    if not tokens_gen and not tokens_ref:
        return 1.0
    if not tokens_gen or not tokens_ref:
        return 0.0
    intersection = tokens_gen.intersection(tokens_ref)
    union = tokens_gen.union(tokens_ref)
    return round(len(intersection) / len(union), 4)


def calculate_emoji_consistency(gen_text: str, ref_text: str) -> float:
    """
    Checks if emoji usage in generated response matches presence in reference response.
    Returns 1.0 if both have emojis or both have none; 0.0 if there is mismatch.
    """
    gen_has_emoji = bool(EMOJI_REGEX.search(gen_text))
    ref_has_emoji = bool(EMOJI_REGEX.search(ref_text))
    return 1.0 if gen_has_emoji == ref_has_emoji else 0.0


def calculate_punctuation_alignment(gen_text: str, ref_text: str) -> float:
    """
    Checks if question mark and exclamation style matches.
    """
    gen_q = "?" in gen_text
    ref_q = "?" in ref_text
    gen_excl = "!" in gen_text
    ref_excl = "!" in ref_text

    score = 0.0
    if gen_q == ref_q:
        score += 0.5
    if gen_excl == ref_excl:
        score += 0.5
    return score
