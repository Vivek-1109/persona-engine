#!/usr/bin/env python3
"""
Persona Engine — Stage 5: Behavioral Features Extractor
Extracts deterministic linguistic and behavioral metadata from conversational persona responses:
- language (hinglish, hindi, english, mixed, unknown)
- tone (casual, humorous, teasing, uncertain, supportive, serious, neutral)
- response_type (answer, question, acknowledgement, refusal, reaction, greeting, invitation, suggestion, statement)
- response_length (character count, word count, length category)
- emoji_count (integer count of unicode emojis)
- has_emoji (boolean flag)
- has_slang (boolean flag for Hinglish/informal slang)
- is_question (boolean flag)
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata


@dataclass
class ResponseLengthInfo:
    char_count: int
    word_count: int
    category: str  # "very_short", "short", "medium", "long"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BehavioralFeatures:
    language: str
    tone: str
    response_type: str
    response_length: ResponseLengthInfo
    emoji_count: int
    has_emoji: bool
    has_slang: bool
    is_question: bool

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


def extract_emojis(text: str) -> List[str]:
    """Extracts all Unicode emoji characters from a string."""
    found: List[str] = []
    for char in text:
        cp = ord(char)
        if (
            unicodedata.category(char) in ("So", "Sk")
            or 0x1F300 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or 0x1F600 <= cp <= 0x1F64F
            or 0x1F680 <= cp <= 0x1F6FF
            or 0x2B50 <= cp <= 0x2B55
        ):
            found.append(char)
    return found


def count_emojis(text: str) -> int:
    """Counts the total number of Unicode emoji characters."""
    return len(extract_emojis(text))


class BehavioralFeatureExtractor:
    """
    Deterministic rule-based extractor for behavioral and stylistic signals.
    Does NOT depend on external LLMs or network APIs.
    """

    # Comprehensive Romanized Hindi / Hinglish lexicon
    HINGLISH_WORDS: Set[str] = {
        "kya", "nhi", "ni", "nahi", "hai", "h", "kar", "kr", "karna", "raha", "rahe",
        "bhai", "bhaiya", "bc", "bsdk", "bencho", "saale", "yaar", "yr", "acha", "achha",
        "thik", "theek", "sahi", "chal", "dekh", "dekha", "bata", "bol", "kaise", "kyu", "kyun",
        "mai", "main", "tu", "tere", "mera", "meri", "mere", "tera", "teri", "ho", "ye",
        "yeh", "wo", "woh", "ab", "kuch", "pe", "par", "se", "ko", "ka", "ki", "ke",
        "tha", "thi", "the", "aaja", "jaa", "aaya", "gaya", "gyi", "wala", "wali",
        "wale", "kab", "kaha", "kahan", "bhi", "to", "toh", "mat", "matt", "aur", "orr",
        "hum", "apna", "apni", "le", "de", "kisko", "usse", "isse", "unhe", "hota", "hoti",
        "hote", "sirf", "bass", "sab", "sabhi", "waise", "wahi", "yahi", "karega", "khelbu",
        "dekhu", "khelu", "jaunga", "aaunga", "bolu", "karu", "raat", "din", "subha", "kal",
        "aaj", "khelega", "nikal", "pohoch", "ruk", "suno", "sun", "batata", "lauda", "chutiya"
    }

    # Core English vocabulary
    ENGLISH_WORDS: Set[str] = {
        "the", "to", "and", "a", "in", "is", "it", "you", "that", "he", "was", "for",
        "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from",
        "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we",
        "when", "your", "can", "said", "there", "use", "an", "each", "which", "she",
        "do", "how", "their", "if", "will", "up", "other", "about", "out", "many", "then",
        "them", "these", "so", "some", "her", "would", "make", "like", "him", "into",
        "time", "has", "look", "two", "more", "write", "go", "see", "number", "no", "way",
        "could", "people", "my", "than", "first", "water", "been", "call", "who", "oil",
        "its", "now", "find", "game", "movie", "phone", "check", "bro", "download", "link",
        "yes", "yeah", "ok", "okay", "done", "wait", "network", "offline", "online", "cool",
        "submit", "assignment", "battery", "laptop", "match"
    }

    # Hinglish & colloquial slang markers
    SLANG_MARKERS: Set[str] = {
        "bc", "bsdk", "bencho", "saale", "chutiya", "lawda", "lauda", "pagle", "katai",
        "chatai", "chud", "chud gyi", "jhol", "faad", "ghanta", "aukaat", "aukat",
        "scene", "bhai", "yaar", "yr", "bro", "lol", "lmao", "rofl", "mast", "badhiya",
        "jugad", "jugaad", "bhag", "terese", "kutta", "kamina"
    }

    # Humorous emojis and tokens
    LAUGH_EMOJIS: Set[str] = {"😂", "🤣", "😆", "😹", "💀", "😭"}



    @staticmethod
    def count_emojis(text: str) -> int:
        """Counts the total number of Unicode emoji characters."""
        return count_emojis(text)

    def detect_language(self, text: str) -> str:
        """
        Classifies language as: 'hindi', 'english', 'hinglish', 'mixed', or 'unknown'.
        """
        has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
        words = re.findall(r"[a-zA-Z]+", text.lower())

        if has_devanagari and not words:
            return "hindi"
        if has_devanagari and words:
            return "mixed"
        if not words:
            return "unknown"

        hinglish_hits = sum(1 for w in words if w in self.HINGLISH_WORDS)
        english_hits = sum(1 for w in words if w in self.ENGLISH_WORDS)

        if hinglish_hits > 0 and english_hits > 0:
            if hinglish_hits >= english_hits:
                return "hinglish"
            return "mixed"
        elif hinglish_hits > 0:
            return "hinglish"
        elif english_hits > 0:
            return "english"

        return "hinglish"

    def detect_tone(self, text: str, has_emoji: bool) -> str:
        """
        Detects conversational tone: casual, humorous, teasing, uncertain, supportive, serious, neutral.
        """
        lower = text.lower()

        # 1. Humor
        if any(e in text for e in self.LAUGH_EMOJIS) or any(
            w in lower for w in ["lol", "lmao", "haha", "rofl", "chud gyi", "maze", "mazee"]
        ):
            return "humorous"

        # 2. Teasing / Banter
        if any(w in lower for w in ["saale", "bsdk", "bencho", "chutiya", "lawda", "lauda", "pagle", "bhag"]):
            return "teasing"

        # 3. Uncertainty
        if any(w in lower for w in ["shayad", "maybe", "pata nhi", "ptani", "dekhna padega", "dekhta hu", "kya pata"]):
            return "uncertain"

        # 4. Supportive
        if any(w in lower for w in ["tension mat", "ho jayega", "ho jaega", "badhiya", "mast", "congrats", "badhai"]):
            return "supportive"

        # 5. Serious
        if any(w in lower for w in ["death", "emergency", "hospital", "police", "fir", "accident", "rip", "serious"]):
            return "serious"

        # 6. Neutral (very brief, direct answers without tone markers)
        if len(text.strip()) <= 4 and not has_emoji:
            return "neutral"

        return "casual"

    def detect_response_type(self, text: str, preceding_user_text: str = "") -> str:
        """
        Detects response type: answer, question, acknowledgement, refusal, reaction, greeting, invitation, suggestion, statement.
        """
        stripped = text.strip()
        lower = stripped.lower()

        # Pure reaction
        if self.count_emojis(stripped) > 0 and not re.search(r"[a-zA-Z0-9]", stripped):
            return "reaction"
        if stripped in {".", "..", "...", "!", "?", "?!", ":)"}:
            return "reaction"

        # Question
        if "?" in stripped or lower.startswith(("kya ", "kyu ", "kyun ", "kab ", "kaha ", "kahan ", "kaise ", "why ", "what ")):
            return "question"

        # Acknowledgement
        ack_tokens = {
            "haa", "haan", "ha", "hn", "haa bhai", "haan bhai", "ok", "okh", "okhh",
            "okk", "sahi h", "sahi hai", "thik", "thik h", "theek", "theek h", "theek hai",
            "thik hai", "theek hai bhai", "thik hai bhai", "ha theek hai", "ha theek hai bhai",
            "haan theek hai", "haan theek hai bhai", "acha", "achha", "achha bhai",
            "hmm", "done", "done deal", "cool", "got it", "yup", "yes", "wahi", "sahi"
        }
        if lower in ack_tokens:
            return "acknowledgement"

        # Refusal
        refusal_tokens = {
            "nhi", "ni", "nahi", "nah", "no", "nope", "nhi bhai", "ni bhai",
            "kabhi nhi", "na", "mat kar", "nhi yaar"
        }
        if lower in refusal_tokens or lower.startswith(("nhi ", "ni ", "nahi ")):
            return "refusal"

        # Greeting / Invitation
        if lower in {"hi", "hello", "hey", "yo", "sup"}:
            return "greeting"
        if lower in {"aaja", "aaja bhai", "chal", "chalo", "aa ja", "aao", "aaja room pe", "chale"}:
            return "invitation"

        # Suggestion
        if any(w in lower for w in ["kar le", "kar lo", "krle", "dekh le", "ye kar", "aise kar", "link se", "bhej diya"]):
            return "suggestion"

        # Statement / Answer
        if len(stripped.split()) >= 6:
            return "statement"

        return "answer"

    def extract_response_length(self, text: str) -> ResponseLengthInfo:
        """
        Calculates character count, word count, and length categorization.
        """
        stripped = text.strip()
        char_count = len(stripped)
        words = stripped.split()
        word_count = len(words)

        if char_count <= 5 or word_count <= 1:
            category = "very_short"
        elif char_count <= 20:
            category = "short"
        elif char_count <= 60:
            category = "medium"
        else:
            category = "long"

        return ResponseLengthInfo(
            char_count=char_count,
            word_count=word_count,
            category=category,
        )

    def has_slang(self, text: str) -> bool:
        """
        Checks whether the text contains informal slang or colloquial youth markers.
        """
        lower = text.lower()
        words = set(re.findall(r"[a-zA-Z]+", lower))

        if any(w in words for w in self.SLANG_MARKERS):
            return True

        # Check multi-word slang phrases
        for phrase in ["chud gyi", "katai chatai", "apna scene", "kuch jhol"]:
            if phrase in lower:
                return True

        return False

    def is_question(self, text: str) -> bool:
        """
        Determines whether the response functions as a question.
        """
        stripped = text.strip().lower()
        if "?" in stripped:
            return True
        if stripped.startswith(("kya ", "kyu ", "kyun ", "kab ", "kaha ", "kahan ", "kaise ", "why ", "what ")):
            return True
        return False

    def extract_features(self, text: str, preceding_user_text: str = "") -> BehavioralFeatures:
        """
        Extracts all behavioral signals for a target response.
        """
        stripped = text.strip()
        emoji_count = self.count_emojis(stripped)
        has_emoji = emoji_count > 0
        lang = self.detect_language(stripped)
        tone = self.detect_tone(stripped, has_emoji)
        resp_type = self.detect_response_type(stripped, preceding_user_text)
        length_info = self.extract_response_length(stripped)
        slang_flag = self.has_slang(stripped)
        question_flag = self.is_question(stripped)

        return BehavioralFeatures(
            language=lang,
            tone=tone,
            response_type=resp_type,
            response_length=length_info,
            emoji_count=emoji_count,
            has_emoji=has_emoji,
            has_slang=slang_flag,
            is_question=question_flag,
        )
