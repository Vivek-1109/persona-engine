"""
Topic Analyzer for Persona Engine Context Engine.
Analyzes full recent conversation window to identify the dominant conversational domain.
Prevents short ambiguous tokens from drowning out strong preceding topic context.
"""

import re
from typing import Any, Dict, List, Tuple

from ml.src.context.context_schema import TopicCategory


class TopicAnalyzer:
    """Deterministic topic classification across multi-turn context."""

    TOPIC_LEXICON: Dict[TopicCategory, Dict[str, float]] = {
        TopicCategory.GAMING: {
            "bgmi": 3.5, "valorant": 3.5, "pubg": 3.5, "gta": 3.0, "granny": 3.0,
            "fifa": 3.5, "minecraft": 3.0, "csgo": 3.5, "steam": 2.5, "game": 2.5,
            "khelega": 3.5, "khelenge": 3.5, "khele": 2.2, "khel": 1.8, "rank": 2.0,
            "squad": 2.5, "noob": 2.0, "clutch": 2.0, "revive": 2.0, "ping": 2.0,
            "fps": 2.0, "login": 1.5, "server": 1.5, "custom": 1.5, "bot": 1.2
        },
        TopicCategory.COLLEGE: {
            "college": 3.0, "attendance": 3.0, "dean": 3.0, "sir": 1.5, "prof": 2.5,
            "professor": 2.5, "class": 2.0, "lecture": 2.0, "exam": 2.5, "admit card": 3.0,
            "library": 2.5, "assignment": 2.5, "submit": 1.5, "submission": 2.0, "lab": 2.0,
            "marks": 2.5, "semester": 2.5, "sem": 2.0, "viva": 2.5, "practical": 2.0,
            "proxy": 3.0, "campus": 2.0, "notice": 2.0, "hod": 2.5
        },
        TopicCategory.TECHNOLOGY: {
            "laptop": 3.0, "processor": 3.0, "ryzen": 3.0, "intel": 2.5, "i5": 3.0,
            "i7": 3.0, "i9": 3.0, "ram": 2.5, "ssd": 2.5, "nitro": 3.0, "acer": 3.0,
            "lenovo": 2.5, "battery": 2.5, "backup": 2.0, "charging": 2.0, "charger": 2.0,
            "phone": 2.0, "android": 2.5, "ios": 2.5, "macbook": 3.0, "windows": 2.5,
            "download": 1.5, "install": 1.5, "update": 1.5, "code": 2.0, "wifi": 2.0
        },
        TopicCategory.PLANS: {
            "canteen": 3.0, "milte": 2.5, "milna": 2.5, "chalte": 2.2, "nikalte": 2.5,
            "nikal": 1.8, "baje": 1.8, "ghoomne": 2.5, "bahar": 2.0, "aaja": 0.5,
            "aana": 0.8, "pohoch": 2.0, "pohochte": 2.0, "raaste": 2.0, "chai": 2.0,
            "tapri": 2.5, "plan": 2.0, "kab tak": 1.8, "aao": 0.8
        },
        TopicCategory.MOVIES: {
            "movie": 3.0, "film": 3.0, "cinema": 3.0, "theater": 3.0, "trailer": 2.5,
            "netflix": 2.5, "scene": 1.5, "actor": 2.0, "heroine": 2.0, "ticket": 2.5,
            "show": 1.8, "series": 2.0, "bollywood": 2.5, "hollywood": 2.5
        },
        TopicCategory.SPORTS: {
            "cricket": 3.0, "kohli": 3.0, "rohit": 2.5, "dhoni": 2.5, "ipl": 3.0,
            "football": 3.0, "messi": 3.0, "ronaldo": 3.0, "goal": 2.5, "wicket": 2.5,
            "score": 1.8, "toss": 2.0
        },
        TopicCategory.SOCIAL: {
            "birthday": 3.0, "party": 3.0, "treat": 2.5, "shaadi": 2.5, "celebration": 2.5,
            "dost": 1.2, "bhai log": 1.5
        },
        TopicCategory.CASUAL_CHAT: {
            "bore": 2.0, "scene": 1.0, "kya scene": 2.5, "kya kar raha": 2.0, "khana": 1.8,
            "dinner": 1.5, "lunch": 1.5, "nashta": 1.5, "neend": 1.8, "so gaya": 1.8,
            "uth gaya": 1.8, "haal": 1.5, "chill": 1.5, "mast": 1.0, "sahi": 0.8
        }
    }

    @classmethod
    def analyze(cls, messages: List[Dict[str, str]]) -> TopicCategory:
        """
        Classifies the dominant topic over the supplied conversation window.
        Recent messages receive higher weight; preceding context anchors terse turns.
        """
        if not messages:
            return TopicCategory.CASUAL_CHAT

        scores: Dict[TopicCategory, float] = {cat: 0.0 for cat in TopicCategory}
        n_msgs = len(messages)

        for idx, turn in enumerate(messages):
            content = turn.get("content", "").lower()
            if not content.strip():
                continue

            # Recency weighting: 1.0 for oldest turn up to 3.0 for newest turn
            is_latest = (idx == n_msgs - 1)
            recency_weight = 1.0 + (idx / max(1, n_msgs - 1)) * 2.0
            if is_latest:
                recency_weight *= 1.2

            # Evaluate each category
            for cat, keywords in cls.TOPIC_LEXICON.items():
                for kw, kw_weight in keywords.items():
                    # Word boundary search
                    pattern = r"\b" + re.escape(kw) + r"\b"
                    if re.search(pattern, content):
                        scores[cat] += kw_weight * recency_weight

        # Pick highest scoring category
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]

        if best_score < 1.0:
            return TopicCategory.CASUAL_CHAT

        return best_cat
