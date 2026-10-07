"""
Persona Engine — Conversational Dataset Analyzer
Analyzes communication styles, speaker turn balances, message lengths,
emoji patterns, Hinglish vocabulary, exclamation/question rates, and n-grams.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..data.loader import load_jsonl

# Unicode range pattern covering popular emojis
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50\u2b55\u200d\ufe0f]",
    flags=re.UNICODE,
)

# Common Romanized Hinglish vocabulary markers
HINGLISH_WORDS = {
    "kya", "hai", "nahi", "bhai", "yaar", "kar", "raha", "rahi", "hu", "hoon",
    "hota", "hoti", "toh", "aur", "bhi", "yeh", "woh", "kaise", "kab", "kaha",
    "accha", "achha", "theek", "sahi", "chal", "dekh", "sun", "arre", "arey",
    "haan", "na", "matlab", "sab", "kuch", "bohot", "bahut", "tera", "mera",
    "apna", "khud", "waise", "waisebhi", "scene", "jugaad", "mast", "chill"
}


@dataclass
class AnalysisReport:
    total_conversations: int = 0
    total_messages: int = 0
    messages_by_speaker: Dict[str, int] = field(default_factory=dict)
    avg_conversation_length: float = 0.0
    median_conversation_length: float = 0.0

    # Message lengths (in words & chars)
    avg_message_length_chars: float = 0.0
    median_message_length_chars: float = 0.0
    avg_message_length_words: float = 0.0
    median_message_length_words: float = 0.0
    speaker_avg_lengths: Dict[str, float] = field(default_factory=dict)

    # Stylistic signatures
    emoji_total_count: int = 0
    emoji_message_ratio: float = 0.0
    top_emojis: List[Tuple[str, int]] = field(default_factory=list)
    question_rate: float = 0.0  # percentage of messages with '?'
    exclamation_rate: float = 0.0  # percentage of messages with '!'

    # Language and mixing
    hinglish_detected_messages: int = 0
    hinglish_ratio: float = 0.0

    # Vocabulary & phrases
    total_words: int = 0
    unique_words: int = 0
    top_words: List[Tuple[str, int]] = field(default_factory=list)
    top_bigrams: List[Tuple[str, int]] = field(default_factory=list)

    # Topics (if annotated)
    topics_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        lines = [
            "==================================================",
            "PERSONA ENGINE — DATASET ANALYSIS SUMMARY",
            "==================================================",
            f"Total Conversations : {self.total_conversations}",
            f"Total Messages      : {self.total_messages}",
            f"Messages by Speaker : {self.messages_by_speaker}",
            f"Avg Conversation Len: {self.avg_conversation_length:.1f} messages",
            "--------------------------------------------------",
            "MESSAGE LENGTHS",
            f"  Avg Chars / Msg   : {self.avg_message_length_chars:.1f} (median: {self.median_message_length_chars:.1f})",
            f"  Avg Words / Msg   : {self.avg_message_length_words:.1f} (median: {self.median_message_length_words:.1f})",
            f"  By Speaker (Chars): {self.speaker_avg_lengths}",
            "--------------------------------------------------",
            "STYLE & PUNCTUATION",
            f"  Question Rate     : {self.question_rate:.1%}",
            f"  Exclamation Rate  : {self.exclamation_rate:.1%}",
            f"  Emoji Frequency   : {self.emoji_message_ratio:.1%} of messages contain emojis",
            f"  Top Emojis        : {[f'{e} ({c})' for e, c in self.top_emojis[:5]]}",
            "--------------------------------------------------",
            "LANGUAGE & VOCABULARY",
            f"  Hinglish Indicator: {self.hinglish_ratio:.1%} messages contain Hinglish markers",
            f"  Total Words       : {self.total_words} (unique: {self.unique_words})",
            f"  Top Words         : {[f'{w} ({c})' for w, c in self.top_words[:8]]}",
            f"  Top Bigrams       : {[f'{b} ({c})' for b, c in self.top_bigrams[:5]]}",
            "==================================================",
        ]
        return "\n".join(lines)


class DatasetAnalyzer:
    """
    Analyzes conversations to extract stylistic and structural metrics.
    """

    def __init__(self, target_speaker: str = "persona"):
        self.target_speaker = target_speaker.lower()

    def analyze(self, conversations: List[Dict[str, Any]]) -> AnalysisReport:
        if not conversations:
            return AnalysisReport()

        total_conversations = len(conversations)
        total_messages = 0
        messages_by_speaker: Counter[str] = Counter()
        conv_lengths: List[int] = []

        msg_char_lens: List[int] = []
        msg_word_lens: List[int] = []
        speaker_chars: Dict[str, List[int]] = {}

        all_emojis: Counter[str] = Counter()
        messages_with_emoji = 0
        questions_count = 0
        exclamations_count = 0
        hinglish_count = 0

        words_counter: Counter[str] = Counter()
        bigrams_counter: Counter[str] = Counter()
        topics_counter: Counter[str] = Counter()

        for conv in conversations:
            messages = conv.get("messages", [])
            conv_lengths.append(len(messages))

            for msg in messages:
                if not isinstance(msg, dict):
                    continue

                total_messages += 1
                speaker = str(msg.get("speaker", "unknown")).lower()
                text = str(msg.get("text", "")).strip()
                messages_by_speaker[speaker] += 1

                # Lengths
                c_len = len(text)
                words = re.findall(r"\b\w+\b", text.lower())
                w_len = len(words)

                msg_char_lens.append(c_len)
                msg_word_lens.append(w_len)
                speaker_chars.setdefault(speaker, []).append(c_len)

                # Stylistic markers
                if "?" in text:
                    questions_count += 1
                if "!" in text:
                    exclamations_count += 1

                emojis_found = EMOJI_PATTERN.findall(text)
                if emojis_found:
                    messages_with_emoji += 1
                    for em in emojis_found:
                        all_emojis[em] += 1

                # Hinglish detection
                has_hinglish = any(w in HINGLISH_WORDS for w in words)
                if has_hinglish:
                    hinglish_count += 1

                # Word & n-gram counters
                for w in words:
                    if len(w) > 1:
                        words_counter[w] += 1

                for i in range(len(words) - 1):
                    bg = f"{words[i]} {words[i+1]}"
                    bigrams_counter[bg] += 1

                # Optional annotations
                ann = msg.get("annotations")
                if isinstance(ann, dict) and "topic" in ann and ann["topic"]:
                    topics_counter[str(ann["topic"])] += 1

        # Aggregations
        speaker_avgs = {
            spk: float(round(statistics.mean(lens), 1))
            for spk, lens in speaker_chars.items()
            if lens
        }

        return AnalysisReport(
            total_conversations=total_conversations,
            total_messages=total_messages,
            messages_by_speaker=dict(messages_by_speaker),
            avg_conversation_length=round(statistics.mean(conv_lengths), 2) if conv_lengths else 0.0,
            median_conversation_length=float(statistics.median(conv_lengths)) if conv_lengths else 0.0,
            avg_message_length_chars=round(statistics.mean(msg_char_lens), 1) if msg_char_lens else 0.0,
            median_message_length_chars=float(statistics.median(msg_char_lens)) if msg_char_lens else 0.0,
            avg_message_length_words=round(statistics.mean(msg_word_lens), 1) if msg_word_lens else 0.0,
            median_message_length_words=float(statistics.median(msg_word_lens)) if msg_word_lens else 0.0,
            speaker_avg_lengths=speaker_avgs,
            emoji_total_count=sum(all_emojis.values()),
            emoji_message_ratio=round(messages_with_emoji / max(1, total_messages), 4),
            top_emojis=all_emojis.most_common(10),
            question_rate=round(questions_count / max(1, total_messages), 4),
            exclamation_rate=round(exclamations_count / max(1, total_messages), 4),
            hinglish_detected_messages=hinglish_count,
            hinglish_ratio=round(hinglish_count / max(1, total_messages), 4),
            total_words=sum(words_counter.values()),
            unique_words=len(words_counter),
            top_words=words_counter.most_common(20),
            top_bigrams=bigrams_counter.most_common(10),
            topics_distribution=dict(topics_counter),
        )

    def analyze_file(self, file_path: Union[str, Path]) -> AnalysisReport:
        records = load_jsonl(file_path)
        return self.analyze(records)


def analyze_dataset(
    data: Union[str, Path, List[Dict[str, Any]]],
    target_speaker: str = "persona",
) -> AnalysisReport:
    """Convenience helper to analyze a dataset file or list."""
    analyzer = DatasetAnalyzer(target_speaker=target_speaker)
    if isinstance(data, (str, Path)):
        return analyzer.analyze_file(data)
    return analyzer.analyze(data)
