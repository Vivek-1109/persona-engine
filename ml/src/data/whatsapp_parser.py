#!/usr/bin/env python3
"""
Persona Engine — WhatsApp Chat Export Parser
Parses raw WhatsApp chat export files into structured, chronological message records.
Handles multiple date/time formats, multiline messages, media tags, deleted messages,
system messages, emojis, URLs, and Hinglish text while preserving original styling.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union
import unicodedata


@dataclass
class ParsedMessage:
    """Represents a single parsed message from a WhatsApp export."""
    timestamp: str  # ISO-formatted or normalized string: "YYYY-MM-DD HH:MM:SS"
    raw_timestamp: str  # Raw timestamp string from the export
    speaker: str  # Speaker name or "SYSTEM"
    text: str  # Full message body (multiline preserved)
    message_type: str  # "text", "media", "deleted", "system", "url", "empty"
    line_number: int  # Source file line number where this message started
    is_persona: bool = False  # Flagged if speaker matches configured persona
    contains_url: bool = False
    contains_emoji: bool = False
    contains_devanagari: bool = False


class WhatsAppParser:
    """
    Parser for WhatsApp chat export text files.
    Robust against diverse locale formats, whitespace variations, and multiline chats.
    """

    # Common timestamp patterns
    # Pattern 1: M/D/YY, H:MM AM/PM - Speaker: Text (US/Standard iOS/Android with \u202f or regular space)
    # Pattern 2: DD/MM/YYYY, HH:MM - Speaker: Text
    # Pattern 3: [DD/MM/YY, HH:MM:SS] Speaker: Text
    # Pattern 4: YYYY-MM-DD, HH:MM - Speaker: Text
    TIMESTAMP_PATTERNS = [
        # Standard: 2/14/25, 5:58 PM - or 14/02/2025, 17:58 -
        re.compile(
            r"^(\d{1,4}[/\.-]\d{1,2}[/\.-]\d{1,4}),?\s+"
            r"(\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202f\u00a0]*(?:[AaPp][Mm]))?)\s*"
            r"[-–—]\s*(.*)$"
        ),
        # Bracketed: [14/02/25, 17:58:12] Name: Text
        re.compile(
            r"^\[(\d{1,4}[/\.-]\d{1,2}[/\.-]\d{1,4}),?\s+"
            r"(\d{1,2}:\d{2}(?::\d{2})?(?:[\s\u202f\u00a0]*(?:[AaPp][Mm]))?)\]\s*(.*)$"
        ),
    ]

    # Media omitted patterns
    MEDIA_PATTERNS = [
        re.compile(r"^<Media omitted>$", re.IGNORECASE),
        re.compile(r"^image omitted$", re.IGNORECASE),
        re.compile(r"^video omitted$", re.IGNORECASE),
        re.compile(r"^audio omitted$", re.IGNORECASE),
        re.compile(r"^sticker omitted$", re.IGNORECASE),
        re.compile(r"^GIF omitted$", re.IGNORECASE),
        re.compile(r"^document omitted$", re.IGNORECASE),
        re.compile(r"^Contact card omitted$", re.IGNORECASE),
        re.compile(r"^Location: https://maps\.google\.com", re.IGNORECASE),
    ]

    # Deleted message patterns
    DELETED_PATTERNS = [
        re.compile(r"^This message was deleted$", re.IGNORECASE),
        re.compile(r"^You deleted this message$", re.IGNORECASE),
    ]

    # System notice patterns
    SYSTEM_PATTERNS = [
        re.compile(r"Messages and calls are end-to-end encrypted", re.IGNORECASE),
        re.compile(r"^Waiting for this message", re.IGNORECASE),
        re.compile(r"security code changed", re.IGNORECASE),
        re.compile(r"changed their phone number", re.IGNORECASE),
        re.compile(r"changed the subject to", re.IGNORECASE),
        re.compile(r"created group", re.IGNORECASE),
        re.compile(r"added you", re.IGNORECASE),
    ]

    URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

    def __init__(self, persona_speaker: Optional[str] = None):
        """
        Initialize the WhatsApp parser.
        
        Args:
            persona_speaker: Optional target persona display name (e.g. "Vivek Jha😎")
        """
        self.persona_speaker = persona_speaker
        self.parsing_errors: List[Dict[str, Union[int, str]]] = []
        self.unparsed_lines: List[Tuple[int, str]] = []

    @staticmethod
    def has_emoji(text: str) -> bool:
        """Check if text contains any emoji character."""
        for char in text:
            # Emoji Unicode blocks & symbol categories
            cp = ord(char)
            if (
                unicodedata.category(char) in ("So", "Sk")
                or 0x1F300 <= cp <= 0x1FAD6
                or 0x2600 <= cp <= 0x27BF
                or 0x1F600 <= cp <= 0x1F64F
                or 0x1F680 <= cp <= 0x1F6FF
            ):
                return True
        return False

    @staticmethod
    def has_devanagari(text: str) -> bool:
        """Check if text contains Devanagari Unicode characters (Hindi script)."""
        return bool(re.search(r"[\u0900-\u097F]", text))

    def parse_datetime(self, date_str: str, time_str: str) -> Optional[datetime]:
        """
        Attempt to parse date and time strings using known WhatsApp timestamp formats.
        """
        # Normalize narrow no-break space and non-breaking space
        clean_time = re.sub(r"[\s\u202f\u00a0]+", " ", time_str).strip()
        combined = f"{date_str.strip()} {clean_time}"

        formats = [
            # 12-hour AM/PM formats
            "%m/%d/%y %I:%M %p",
            "%m/%d/%Y %I:%M %p",
            "%d/%m/%y %I:%M %p",
            "%d/%m/%Y %I:%M %p",
            "%Y-%m-%d %I:%M %p",
            # 24-hour formats
            "%d/%m/%Y %H:%M",
            "%d/%m/%y %H:%M",
            "%m/%d/%Y %H:%M",
            "%m/%d/%y %H:%M",
            "%Y-%m-%d %H:%M",
            # With seconds
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%y %H:%M:%S",
            "%m/%d/%y %I:%M:%S %p",
            "%m/%d/%Y %I:%M:%S %p",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(combined, fmt)
            except ValueError:
                continue

        return None

    def classify_message_type(self, speaker: str, text: str) -> str:
        """Categorize message into type: text, media, deleted, system, url, empty."""
        text_clean = text.strip()
        if speaker == "SYSTEM":
            return "system"
        if not text_clean:
            return "empty"
        for pat in self.MEDIA_PATTERNS:
            if pat.match(text_clean):
                return "media"
        for pat in self.DELETED_PATTERNS:
            if pat.match(text_clean):
                return "deleted"
        for pat in self.SYSTEM_PATTERNS:
            if pat.search(text_clean):
                return "system"
        # Pure URL check: entire text is a URL with no other conversational content
        urls = self.URL_REGEX.findall(text_clean)
        if len(urls) == 1 and text_clean == urls[0]:
            return "url"
        return "text"

    def parse_lines(self, lines: List[str]) -> List[ParsedMessage]:
        """
        Parse raw lines of WhatsApp export into a list of ParsedMessage objects.
        Handles multiline continuation seamlessly.
        """
        messages: List[ParsedMessage] = []
        current_msg: Optional[Dict] = None

        def finalize_message(msg_dict: Dict) -> ParsedMessage:
            speaker = msg_dict["speaker"]
            raw_text = msg_dict["text"]
            msg_type = self.classify_message_type(speaker, raw_text)
            is_persona = bool(self.persona_speaker and speaker == self.persona_speaker)
            urls = self.URL_REGEX.findall(raw_text)

            return ParsedMessage(
                timestamp=msg_dict["timestamp"],
                raw_timestamp=msg_dict["raw_timestamp"],
                speaker=speaker,
                text=raw_text,
                message_type=msg_type,
                line_number=msg_dict["line_number"],
                is_persona=is_persona,
                contains_url=len(urls) > 0,
                contains_emoji=self.has_emoji(raw_text),
                contains_devanagari=self.has_devanagari(raw_text),
            )

        for line_idx, line in enumerate(lines):
            line_no = line_idx + 1
            line_clean = line.rstrip("\r\n")

            matched = False
            for pat in self.TIMESTAMP_PATTERNS:
                m = pat.match(line_clean)
                if m:
                    matched = True
                    # Finalize previous message
                    if current_msg is not None:
                        messages.append(finalize_message(current_msg))

                    date_str, time_str, rest = m.groups()
                    dt = self.parse_datetime(date_str, time_str)
                    iso_ts = dt.strftime("%Y-%m-%d %H:%M:%S") if dt else f"{date_str} {time_str}"
                    raw_ts = f"{date_str}, {time_str}"

                    # Determine speaker and text
                    if ": " in rest:
                        speaker, text = rest.split(": ", 1)
                    else:
                        speaker = "SYSTEM"
                        text = rest

                    current_msg = {
                        "timestamp": iso_ts,
                        "raw_timestamp": raw_ts,
                        "speaker": speaker,
                        "text": text,
                        "line_number": line_no,
                    }
                    break

            if not matched:
                if current_msg is not None:
                    # Multiline continuation: append with newline
                    current_msg["text"] += "\n" + line_clean
                else:
                    # Stray line before the first timestamp
                    if line_clean.strip():
                        self.unparsed_lines.append((line_no, line_clean))
                        self.parsing_errors.append({
                            "line_number": line_no,
                            "content": line_clean,
                            "error": "Line before first valid timestamp"
                        })

        if current_msg is not None:
            messages.append(finalize_message(current_msg))

        return messages

    def parse_file(self, filepath: Union[str, Path]) -> List[ParsedMessage]:
        """Parse raw WhatsApp export file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"WhatsApp export file not found: {path}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        return self.parse_lines(lines)
