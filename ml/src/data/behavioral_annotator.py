#!/usr/bin/env python3
"""
Persona Engine — Behavioral & Stylistic Annotator
Heuristic rule-based annotation pipeline for conversational persona responses.
Derives metadata on language, tone, response type, length, emojis, and sensitive PII.
Operates deterministically without external API dependencies.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Tuple
import unicodedata


@dataclass
class BehavioralAnnotation:
    """Represents behavioral and linguistic annotations for a target response."""
    language: str  # "hinglish", "hindi", "english", "mixed", "unknown"
    tone: str  # "casual", "humorous", "serious", "neutral", "teasing", "supportive", "uncertain"
    response_type: str  # "answer", "question", "acknowledgement", "suggestion", "refusal", "joke", "reaction", "greeting", "invitation", "information", "unknown"
    length_category: str  # "very_short", "short", "medium", "long"
    emoji_present: bool
    question_present: bool
    exclamation_present: bool
    contains_sensitive_pii: bool
    sensitive_categories: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BehavioralAnnotator:
    """
    Rule-based annotator analyzing personality traits and linguistic signals.
    """

    # Hinglish vocabulary markers
    HINGLISH_WORDS = {
        "kya", "nhi", "ni", "nahi", "hai", "h", "kar", "kr", "karna", "raha", "rahe",
        "bhai", "bhaiya", "bc", "bsdk", "bencho", "saale", "yaar", "yr", "acha", "achha",
        "thik", "theek", "sahi", "chal", "dekh", "dekha", "bata", "bol", "kaise", "kyu", "kyun",
        "mai", "main", "tu", "tere", "mera", "meri", "mere", "tera", "teri", "ho", "ye",
        "yeh", "wo", "woh", "ab", "kuch", "pe", "par", "se", "ko", "ka", "ki", "ke",
        "tha", "thi", "the", "aaja", "jaa", "aaya", "gaya", "gyi", "wala", "wali",
        "wale", "kab", "kaha", "kahan", "bhi", "to", "toh", "mat", "matt", "aur", "orr",
        "hum", "apna", "apni", "le", "de", "kisko", "usse", "isse", "unhe", "hota", "hoti",
        "hote", "sirf", "bass", "sab", "sabhi", "waise", "wahi", "yahi", "karega", "khelbu",
        "dekhu", "khelu", "jaunga", "aaunga", "bolu", "karu", "raat", "din", "subha"
    }

    # Common English words
    ENGLISH_WORDS = {
        "the", "to", "and", "a", "in", "is", "it", "you", "that", "he", "was", "for",
        "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from",
        "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we",
        "when", "your", "can", "said", "there", "use", "an", "each", "which", "she",
        "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then",
        "them", "these", "so", "some", "her", "would", "make", "like", "him", "into",
        "time", "has", "look", "two", "more", "write", "go", "see", "number", "no", "way",
        "could", "people", "my", "than", "first", "water", "been", "call", "who", "oil",
        "its", "now", "find", "game", "movie", "phone", "check", "bro", "download", "link",
        "yes", "yeah", "ok", "okay", "done", "wait", "network", "offline", "online", "cool"
    }

    # Emojis associated with laughter/humor
    LAUGH_EMOJIS = {"😂", "🤣", "😆", "😹", "💀", "😭"}

    # Sensitive PII Patterns
    RE_PHONE = re.compile(r"(?:\+?91[\-\s]?)?[6-9]\d{9}\b")
    RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    RE_UPI = re.compile(r"\b[\w.\-]+@(upi|okhdfcbank|okicici|okaxis|oksbi|paytm)\b", re.IGNORECASE)
    RE_SECRET = re.compile(r"(?i)\b(?:password|pwd|otp|passcode|secret)\s*[:=]\s*\S+")

    @staticmethod
    def has_emoji(text: str) -> bool:
        """Check for emoji presence."""
        for char in text:
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

    def detect_language(self, text: str) -> str:
        """Detect language: hindi, english, hinglish, mixed, unknown."""
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
        words = set(re.findall(r"[a-zA-Z]+", text.lower()))

        if has_devanagari and not words:
            return "hindi"
        if has_devanagari and words:
            return "mixed"

        if not words:
            return "unknown"

        hinglish_matches = sum(1 for w in words if w in self.HINGLISH_WORDS)
        english_matches = sum(1 for w in words if w in self.ENGLISH_WORDS)

        if hinglish_matches > 0 and english_matches > 0:
            if hinglish_matches >= english_matches:
                return "hinglish"
            return "mixed"
        elif hinglish_matches > 0:
            return "hinglish"
        elif english_matches > 0:
            return "english"

        # Fallback based on character distribution
        return "hinglish"

    def detect_tone(self, text: str, has_emoji: bool) -> str:
        """Detect conversational tone."""
        lower = text.lower()

        # Check humor
        if any(e in text for e in self.LAUGH_EMOJIS) or any(
            w in lower for w in ["lol", "lmao", "haha", "rofl", "chud gyi", "maze", "mazee"]
        ):
            return "humorous"

        # Check teasing
        if any(w in lower for w in ["saale", "bsdk", "bencho", "chutiya", "lawda", "pagle", "teri maa", "bhag"]):
            return "teasing"

        # Check uncertainty
        if any(w in lower for w in ["shayad", "maybe", "pata nhi", "ptani", "dekhna padega", "dekhta hu", "kya pata"]):
            return "uncertain"

        # Check supportive
        if any(w in lower for w in ["tension mat", "ho jayega", "ho jaega", "badhiya", "mast", "congrats", "badhai"]):
            return "supportive"

        # Check serious
        if any(w in lower for w in ["death", "emergency", "hospital", "police", "fir", "accident", "rip", "serious"]):
            return "serious"

        # Plain short responses
        if len(text.strip()) <= 4 and not has_emoji:
            return "neutral"

        return "casual"

    def detect_response_type(self, text: str, preceding_user_text: str = "") -> str:
        """Detect response type."""
        stripped = text.strip()
        lower = stripped.lower()

        # Pure reaction (only emoji or punctuation)
        if self.has_emoji(stripped) and not re.search(r"[a-zA-Z0-9]", stripped):
            return "reaction"
        if stripped in {".", "..", "...", "!", "?", "?!"}:
            return "reaction"

        # Question
        if "?" in stripped or lower.startswith(("kya ", "kyu ", "kyun ", "kab ", "kaha ", "kaise ", "why ", "what ")):
            return "question"

        # Acknowledgement
        ack_tokens = {
            "haa", "haan", "ha", "hn", "haa bhai", "haan bhai", "ok", "okh", "okhh",
            "okk", "sahi h", "sahi hai", "thik", "thik h", "theek", "acha", "achha",
            "hmm", "done", "done deal", "cool", "got it", "yup", "yes", "wahi"
        }
        if lower in ack_tokens:
            return "acknowledgement"

        # Refusal
        refusal_tokens = {
            "nhi", "ni", "nahi", "nah", "no", "nope", "nhi bhai", "ni bhai",
            "kabhi nhi", "na", "mat kar", "nhi yaar"
        }
        if lower in refusal_tokens or lower.startswith("nhi ") or lower.startswith("ni "):
            return "refusal"

        # Greeting / Invitation
        if lower in {"hi", "hello", "hey", "yo", "sup"}:
            return "greeting"
        if lower in {"aaja", "aaja bhai", "chal", "chalo", "aa ja", "aao", "aaja room pe"}:
            return "invitation"

        # Suggestion
        if any(w in lower for w in ["kar le", "kar lo", "krle", "dekh le", "ye kar", "aise kar", "link se"]):
            return "suggestion"

        # Default is answer
        return "answer"

    def detect_sensitive_pii(self, text: str) -> Tuple[bool, List[str]]:
        """Identify potential sensitive PII for privacy flagging."""
        flagged: List[str] = []

        if self.RE_PHONE.search(text):
            flagged.append("phone")
        if self.RE_EMAIL.search(text):
            flagged.append("email")
        if self.RE_UPI.search(text):
            flagged.append("upi")
        if self.RE_SECRET.search(text):
            flagged.append("credentials")

        return (len(flagged) > 0, flagged)

    def annotate(self, text: str, preceding_user_text: str = "") -> BehavioralAnnotation:
        """
        Derive full behavioral metadata for a target persona response.
        """
        stripped = text.strip()
        length = len(stripped)

        # Length category
        if length <= 5 or len(stripped.split()) <= 1:
            len_cat = "very_short"
        elif length <= 20:
            len_cat = "short"
        elif length <= 60:
            len_cat = "medium"
        else:
            len_cat = "long"

        has_em = self.has_emoji(stripped)
        has_q = "?" in stripped
        has_excl = "!" in stripped

        lang = self.detect_language(stripped)
        tone = self.detect_tone(stripped, has_em)
        resp_type = self.detect_response_type(stripped, preceding_user_text)
        is_pii, pii_cats = self.detect_sensitive_pii(stripped)

        return BehavioralAnnotation(
            language=lang,
            tone=tone,
            response_type=resp_type,
            length_category=len_cat,
            emoji_present=has_em,
            question_present=has_q,
            exclamation_present=has_excl,
            contains_sensitive_pii=is_pii,
            sensitive_categories=pii_cats,
        )
