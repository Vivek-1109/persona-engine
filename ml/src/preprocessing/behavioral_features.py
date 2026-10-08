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


EMOJI_REGEX = re.compile(
    r"("
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U00002B50-\U00002B55\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]"
    r"(?:[\U0001F3FB-\U0001F3FF]|\uFE0E|\uFE0F)?"
    r"(?:\u200D[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U00002B50-\U00002B55\U0001F600-\U0001F64F\U0001F680-\U0001F6FF](?:[\U0001F3FB-\U0001F3FF]|\uFE0E|\uFE0F)?)*"
    r")"
)


def extract_emojis(text: str) -> List[str]:
    """
    Extracts Unicode emoji sequences from a string.
    Correctly coalesces base emojis + skin tone modifiers (U+1F3FB..U+1F3FF),
    variation selectors, and ZWJ sequences as single emoji grapheme units.
    """
    return EMOJI_REGEX.findall(text)


def count_emojis(text: str) -> int:
    """Counts the total number of Unicode emoji grapheme clusters."""
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
        Uses semantic lexical cues, affect, and stylistic markers rather than arbitrary string length.
        """
        stripped = text.strip()
        lower = stripped.lower()

        # 1. Teasing / Banter (explicit roast, profanity, peer insults)
        teasing_keywords = [
            "saale", "bsdk", "bencho", "chutiya", "lawda", "lauda", "pagle",
            "chomu", "jhaatu", "kutta", "kamina", "bhadwe", "chudega", "chud gyi",
            "teri maa", "gadhe", "andha hai"
        ]
        if any(w in lower for w in teasing_keywords):
            return "teasing"

        # 2. Humor (laugh emojis or explicit humor tokens)
        if any(e in text for e in self.LAUGH_EMOJIS) or any(
            w in lower for w in ["lol", "lmao", "haha", "hahaha", "rofl", "xdd", "xd", "joke", "mazak", "hasna", "comedy"]
        ):
            return "humorous"

        # 3. Uncertainty (hedging, doubt, speculation)
        uncertain_keywords = [
            "shayad", "maybe", "pata nhi", "ptani", "dekhna padega", "dekhta hu",
            "kya pata", "probably", "not sure", "idk", "lagta hai", "doubt", "confirm nahi", "pakka nahi"
        ]
        if any(w in lower for w in uncertain_keywords):
            return "uncertain"

        # 4. Supportive (reassurance, encouragement, cheering, empathy)
        supportive_keywords = [
            "tension mat", "ho jayega", "ho jaega", "badhiya", "mast", "congrats",
            "badhai", "chill kar", "koi na", "koi baat nahi", "all the best", "take care",
            "shabash", "sahi kiya", "help chahiye", "well done", "sambhal lena"
        ]
        if any(w in lower for w in supportive_keywords):
            return "supportive"

        # 5. Serious (emergencies, critical issues, deadlines, high-stakes matters)
        serious_keywords = [
            "death", "emergency", "hospital", "police", "fir", "accident", "rip",
            "serious", "urgent", "zaroori", "jaruri", "deadline", "critical",
            "strictly", "pareshan", "bimaari", "ill", "problem", "strict"
        ]
        if any(w in lower for w in serious_keywords):
            return "serious"

        # 6. Neutral (purely factual, matter-of-fact statements, numerical replies, direct objective answers)
        # Check if text is devoid of informal slang, emojis, exclamations, and affective particles
        is_numeric = bool(re.match(r"^[\d\:\.\s\/\-]+$", stripped))
        has_slang_words = self.has_slang(stripped)
        has_casual_particle = any(
            re.search(rf"\b{re.escape(w)}\b", lower)
            for w in ["bhai", "yaar", "yr", "bro", "chal", "aaja", "na", "re", "arre", "bhaiya", "sahi h"]
        )
        has_exclamation = "!" in stripped

        if is_numeric:
            return "neutral"

        if not has_emoji and not has_slang_words and not has_casual_particle and not has_exclamation:
            # Objective confirmations or factual status remarks without peer markers
            neutral_tokens = {
                "ok", "ok.", "okh", "okhh", "theek", "theek.", "done", "done.",
                "haan", "haan.", "ha", "nahi", "nhi", "yes", "no", "received", "available",
                "sent", "sent.", "offline tha", "online hu", "link bhej diya"
            }
            if lower in neutral_tokens or lower.endswith("."):
                return "neutral"
            # Informational declarative sentences without slang or emotional coloring
            if not any(w in lower for w in ["kya", "kyu", "bc", "bhai", "yaar"]):
                return "neutral"

        return "casual"

    def detect_response_type(self, text: str, preceding_user_text: str = "") -> str:
        """
        Detects response type: answer, question, acknowledgement, refusal, reaction, greeting, invitation, suggestion, statement.
        Determines answer vs statement using conversational context (preceding user inquiry) rather than word length.
        """
        stripped = text.strip()
        lower = stripped.lower()

        # 1. Pure reaction (emojis only or pure punctuation)
        if self.count_emojis(stripped) > 0 and not re.search(r"[a-zA-Z0-9]", stripped):
            return "reaction"
        if stripped in {".", "..", "...", "!", "?", "?!", ":)"}:
            return "reaction"

        # 2. Question
        if self.is_question(stripped):
            return "question"

        # 3. Refusal
        refusal_tokens = {
            "nhi", "ni", "nahi", "nah", "no", "nope", "nhi bhai", "ni bhai",
            "kabhi nhi", "na", "mat kar", "nhi yaar", "nahi aa paunga", "nahi hoga"
        }
        if lower in refusal_tokens or lower.startswith(("nhi ", "ni ", "nahi ", "nah ", "no ")):
            return "refusal"

        # 4. Invitation
        invitation_tokens = {
            "aaja", "aaja bhai", "chal", "chalo", "chale", "aa ja", "aao",
            "aaja room pe", "aaja discord pe", "aaja lobby", "join kar"
        }
        if lower in invitation_tokens or lower.startswith(("aaja ", "chal ", "chalo ")):
            return "invitation"

        # 5. Suggestion / Directive
        suggestion_phrases = [
            "kar le", "kar lo", "krle", "dekh le", "ye kar", "aise kar",
            "link se", "bhej diya", "download kar", "try kar", "bhej de", "bol de", "call kar"
        ]
        if any(w in lower for w in suggestion_phrases) and not lower.startswith(("kya", "kyu")):
            return "suggestion"

        # 6. Acknowledgement
        ack_tokens = {
            "haa", "haan", "ha", "hn", "haa bhai", "haan bhai", "ok", "okh", "okhh",
            "okk", "sahi h", "sahi hai", "thik", "thik h", "theek", "theek h", "theek hai",
            "thik hai", "theek hai bhai", "thik hai bhai", "ha theek hai", "ha theek hai bhai",
            "haan theek hai", "haan theek hai bhai", "ok theek hai", "ok theek hai bhai",
            "ok thik hai", "ok thik hai bhai", "acha", "achha", "achha bhai",
            "hmm", "done", "done deal", "cool", "got it", "yup", "yes", "wahi", "sahi"
        }
        if lower in ack_tokens or (any(lower.startswith(f"{t} ") for t in ["ok", "okh", "haa", "haan", "theek hai", "thik hai"]) and any(w in lower for w in ["theek", "thik", "sahi", "bhai"])):
            return "acknowledgement"

        # 7. Greeting
        if lower in {"hi", "hello", "hey", "yo", "sup"}:
            return "greeting"

        # 8. Answer vs. Statement (Determined by Preceding Context)
        last_u = preceding_user_text.lower().strip()
        is_preceding_inquiry = (
            "?" in last_u
            or any(
                re.search(rf"\b{re.escape(w)}\b", last_u)
                for w in [
                    "kya", "kyu", "kyun", "kab", "kaha", "kahan", "kaise", "kidhar",
                    "kon", "kaun", "kitna", "kitne", "kitni", "why", "what", "when",
                    "where", "how", "who", "which", "khelega", "chalega", "aayega",
                    "free hai", "bata", "batao", "bol", "bheja", "bhejo"
                ]
            )
        )
        is_answer_syntax = lower.startswith(("kyuki", "because", "isliye", "reason"))

        if is_preceding_inquiry or is_answer_syntax:
            return "answer"

        return "statement"

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
