#!/usr/bin/env python3
"""
Persona Engine — Stage 5: Deterministic Topic Classifier
Lightweight, rule-based topic classification for conversational persona utterances.
Categories:
- gaming
- college
- movies
- technology
- plans
- casual_chat
- sports
- social
- other

Operates purely deterministically without external LLMs or APIs.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TopicClassification:
    primary_topic: str
    confidence: float
    scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeterministicTopicClassifier:
    """
    Lightweight keyword- and pattern-driven classifier for conversation topics.
    Evaluates target response content and optionally preceding conversational context.
    """

    TOPICS = [
        "gaming",
        "college",
        "movies",
        "technology",
        "plans",
        "casual_chat",
        "sports",
        "social",
        "other",
    ]

    # Weighted keyword lexicons per category
    KEYWORDS: Dict[str, Dict[str, float]] = {
        "gaming": {
            # Specific titles & game terms (high weight)
            "bgmi": 3.0, "pubg": 3.0, "valorant": 3.0, "val": 2.5, "fifa": 3.0,
            "granny": 3.0, "steam": 2.5, "gta": 3.0, "cod": 2.5, "free fire": 3.0,
            "minecraft": 3.0, "clutch": 2.5, "kd": 2.5, "rank": 2.0, "headshot": 2.5,
            "respawn": 2.5, "lobby": 2.0, "ping": 2.0, "lag": 1.5, "server": 1.5,
            "noob": 2.0, "bot": 1.5, "pro": 1.5, "multiplayer": 2.0, "gameplay": 2.0,
            # Common verbs / actions (medium weight)
            "khelega": 2.5, "khelo": 2.0, "khelte": 2.0, "khela": 2.0, "khel": 1.5,
            "game": 2.0, "match khela": 2.5, "room banaye": 2.5, "custom": 2.0,
        },
        "college": {
            "college": 3.0, "clg": 2.5, "campus": 2.5, "hostel": 2.5, "canteen": 2.0,
            "prof": 2.5, "professor": 2.5, "sir": 1.0, "lecture": 2.5, "attendance": 3.0,
            "exam": 3.0, "test": 2.0, "midsem": 3.0, "endsem": 3.0, "practical": 2.5,
            "assignment": 3.0, "submit": 2.0, "submission": 2.5, "project": 1.5,
            "lab": 2.0, "syllabus": 2.5, "semester": 2.5, "sem": 2.0, "cgpa": 3.0,
            "marks": 2.0, "grade": 2.0, "roll no": 2.5, "admit card": 3.0, "class": 1.5,
            "bunk": 2.5, "placement": 2.5, "internship": 2.5, "library": 2.0,
        },
        "movies": {
            "movie": 3.0, "film": 3.0, "cinema": 3.0, "theatre": 2.5, "theater": 2.5,
            "trailer": 2.5, "teaser": 2.5, "series": 2.0, "web series": 3.0,
            "netflix": 3.0, "prime": 2.0, "hotstar": 2.5, "episode": 2.5, "season": 2.0,
            "actor": 2.0, "actress": 2.0, "director": 2.0, "interval": 2.0, "hall": 1.5,
            "show": 1.5, "nutshell": 2.0, "story": 1.5, "scene": 1.0, "review": 2.0,
            "ott": 2.5, "popcorn": 2.0, "tickets": 2.0, "booking": 1.5,
        },
        "technology": {
            "laptop": 3.0, "phone": 2.5, "mobile": 2.0, "pc": 2.0, "computer": 2.5,
            "processor": 3.0, "ryzen": 3.0, "intel": 2.5, "i5": 2.5, "i7": 2.5,
            "gpu": 3.0, "rtx": 3.0, "graphics card": 3.0, "ram": 2.5, "ssd": 2.5,
            "storage": 2.0, "battery": 2.5, "charger": 2.0, "wifi": 2.0, "bluetooth": 2.0,
            "download": 2.0, "link": 1.5, "code": 2.0, "coding": 2.5, "bug": 2.0,
            "windows": 2.5, "android": 2.5, "ios": 2.5, "iphone": 2.5, "mac": 2.5,
            "nitro": 2.5, "acer": 2.5, "asus": 2.5, "software": 2.0, "update": 1.5,
        },
        "plans": {
            "aaja": 2.0, "chal": 1.5, "chalo": 1.5, "chale": 2.0, "chalega": 2.0,
            "nikal": 2.0, "nikle": 2.0, "nikalte": 2.0, "milte": 2.0, "mil": 1.5,
            "kab": 1.5, "kaha": 1.5, "kahan": 1.5, "plan": 2.5, "scene kya hai": 3.0,
            "weekend": 2.0, "shaam": 1.5, "sham": 1.5, "raat": 1.0, "subah": 1.5,
            "kal": 1.5, "parso": 2.0, "aaj": 1.0, "time pe": 1.5, "baje": 1.5,
            "ghoomne": 2.5, "trip": 2.5, "bahar": 1.5, "chai": 1.5, "tea": 1.5,
        },
        "sports": {
            "cricket": 3.0, "match": 2.5, "ipl": 3.0, "kohli": 3.0, "rohit": 3.0,
            "dhoni": 3.0, "century": 3.0, "wicket": 2.5, "score": 2.0, "over": 1.5,
            "ball": 1.5, "six": 1.5, "four": 1.5, "football": 3.0, "goal": 2.5,
            "messi": 3.0, "ronaldo": 3.0, "badminton": 3.0, "court": 2.0, "toss": 2.0,
            "final": 1.5, "semi final": 2.0, "chase": 2.0, "batting": 2.5, "bowling": 2.5,
            "india": 1.5, "cup": 1.5, "trophy": 2.0,
        },
        "social": {
            "family": 2.5, "mummy": 2.5, "papa": 2.5, "bhaiya": 2.0, "bhabhi": 2.0,
            "didi": 2.0, "ghar": 1.5, "relatives": 2.5, "shadi": 2.5, "shaadi": 2.5,
            "party": 2.0, "treat": 2.0, "birthday": 2.5, "bday": 2.5, "dost": 1.5,
            "friends": 1.5, "group": 1.5, "call kar": 2.0, "phone kiya": 2.0,
            "baat hui": 2.0, "mili thi": 1.5, "mila tha": 1.5,
        },
        "casual_chat": {
            "kya haal": 2.5, "kya chal raha": 2.5, "kuch nhi": 2.0, "kuch nahi": 2.0,
            "bore": 2.0, "sahi h": 2.0, "sahi hai": 2.0, "thik h": 2.0, "theek": 1.5,
            "acha": 1.5, "achha": 1.5, "haha": 2.0, "lol": 2.0, "lmao": 2.0,
            "bencho": 2.0, "bsdk": 2.0, "saale": 1.5, "yaar": 1.5, "suno": 1.5,
            "arre": 1.5, "kya hua": 2.0, "sach me": 2.0, "waise": 1.5, "mast": 1.5,
        },
    }

    def classify(self, text: str, context_text: str = "") -> TopicClassification:
        """
        Classifies the conversation snippet into one of the 9 categories.
        Uses combined text with target response prioritized.
        """
        target_lower = text.lower()
        context_lower = context_text.lower()
        combined_lower = f"{context_lower} {target_lower}".strip()

        scores: Dict[str, float] = {topic: 0.0 for topic in self.TOPICS}

        for topic, lexicon in self.KEYWORDS.items():
            topic_score = 0.0
            for term, weight in lexicon.items():
                # Target message matches get full weight
                if re.search(rf"\b{re.escape(term)}\b", target_lower):
                    topic_score += weight * 1.5
                # Preceding context matches get baseline weight (context awareness)
                elif context_lower and re.search(rf"\b{re.escape(term)}\b", context_lower):
                    topic_score += weight * 0.8

            scores[topic] = round(topic_score, 2)

        # Determine primary topic
        max_score = 0.0
        primary = "other"

        # Check candidate with highest score
        for topic in self.TOPICS:
            if topic == "other":
                continue
            if scores[topic] > max_score:
                max_score = scores[topic]
                primary = topic

        # If no explicit category matched, check if it's casual conversational filler
        if max_score < 1.0:
            if len(text.strip().split()) <= 4 or any(
                w in target_lower for w in ["haan", "ha", "nhi", "ok", "acha", "hmm", "thik", "bhai"]
            ):
                primary = "casual_chat"
                max_score = 1.0
                scores["casual_chat"] = 1.0
            else:
                primary = "other"
                scores["other"] = 1.0
                max_score = 1.0

        confidence = round(min(1.0, max_score / 5.0), 2)
        if confidence < 0.2:
            confidence = 0.2

        return TopicClassification(
            primary_topic=primary,
            confidence=confidence,
            scores=scores,
        )
