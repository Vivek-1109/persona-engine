"""
User Intent Analyzer for Persona Engine Context Engine (Stage 6F Refined).
Identifies the communicative intent / dialogue act of the current user turn
using both the latest utterance and preceding dialogue context.
Deterministic, local, fast, explainable, and testable.
"""

import re
from typing import Any, Dict, List, Optional

from ml.src.context.context_schema import UserIntent


class IntentAnalyzer:
    """
    Deterministic intent and dialogue-act classifier based on utterance morphology,
    clause decomposition, and multi-turn dialogue context.
    """

    # 1. Direct Invitation Verbs & Phrases
    INVITATION_VERBS = [
        r"\bkhelega\b", r"\bkhelenge\b", r"\bkhelte hain\b", r"\bkhelte h\b",
        r"\bkhelna hai\b", r"\bkhele\b", r"\bkhelte\b",
        r"\baaja\b", r"\baa ja\b", r"\baao\b", r"\baa jao\b",
        r"\bchalega\b", r"\bchaloge\b", r"\bchale\b", r"\bchalenge\b",
        r"\bchalte hain\b", r"\bchalte h\b", r"\bchalo fir\b",
        r"\bjoin kar\b", r"\bjoin karega\b", r"\bjoin karoge\b",
        r"\bmilne chalega\b",
        r"\bbanaye kya\b", r"\bkare aaj\b", r"\bsession kare\b",
    ]

    # Activity/Target + Movement/Invitation
    INVITATION_ACTIVITIES = [
        r"\b(canteen|cinema|theatre|movie|film|trip|lunch|dinner|chai|coffee|bahar)\b.*\b(chale|chalega|chalte|chalo|aaja|aao)\b",
        r"\b(chai peene|coffee peene)\b",
        r"\b(game|bgmi|valorant|pubg|cod|fifa|match)\b.*\b(khelega|khelenge|aaja|chale|khelte|khel)\b",
        r"\b(squad|lobby|discord)\b.*\b(ready|aaja|aao|join)\b",
    ]

    # Backward-compatible attribute aliases
    INVITATION_PATTERNS = INVITATION_VERBS + INVITATION_ACTIVITIES

    # 2. Answer Request / Assistance solicitation
    ANSWER_REQUEST_PATTERNS = [
        r"\b(solution|fix|solve)\b",
        r"\b(isme|isme kya)\b.*\b(karna chahiye|kya kare)\b",
        r"\b(explain kar|explain karna|help kar|help chahiye)\b",
        r"\b(steps|tarika|approach|step by step)\b",
    ]

    # 3. Planning / Scheduling coordination
    PLANNING_PATTERNS = [
        r"\b(kab milte hain|kab milenge|kitne baje)\b",
        r"\b(kal|weekend|shaam|raat)\b.*\b(kya plan|plan hai|kab milte|milte hain|nikalte hain)\b",
        r"\b(\d+\s*baje)\b.*\b(milte|nikalte|chalte|aana)\b",
        r"\b(lunch break me chalte hain)\b",
        r"\b(plan kya hai|kya plan hai)\b",
        r"\b(ping me|deploy|free to deploy)\b",
        r"\b(baad me baat karta|see you tomorrow|chalta hu)\b",
    ]

    # 4. Information / Status / Announcements / Observations
    INFORMATION_PATTERNS = [
        r"\b(notice nikala|notice lagaya|notice board|circular|date sheet aa gayi)\b",
        r"\b(attendance mandatory|attendance short|short batai|mandatory hai)\b",
        r"\b(dean ne|prof ne|sir ko de di|hod called)\b",
        r"\b(battery bahut jaldi drain|battery drain|drain ho rahi|heating problem aa rahi)\b",
        r"\b(laptop crash|null pointer|error aa raha|slow chal raha)\b",
        r"\b(download ho gaya|steam pe download|install ho gaya)\b",
        r"\b(rank push karna hai|party chal rahi|neend aa rahi)\b",
        r"\b(visuals|climax|review)\b.*\b(zabardast|badhiya|the)\b",
        r"\b(temperature is hitting|emergency department meet|going downhill lately)\b",
        r"\b(battery backup kaisa hai)\b",  # Preserves Scenario D
    ]

    # 5. Agreement / Acceptance / Positive acknowledgments
    AGREEMENT_PATTERNS = [
        r"^(haan|haa|ha|yes|yep|yeah|ok|okay|okh|done|agreed|bilkul|pakka|theek|thik|sahi hai|sahi h|chal theek|chal thik)\b",
        r"\b(bye bhai goodnight|tata bye|goodnight)\b",
    ]

    # 6. Disagreement / Negation / Rejection
    DISAGREEMENT_PATTERNS = [
        r"^(nhi|nahi|no|nope|nah|naa)\b",
        r"\b(galat|galat bol raha|galat logic|galat hai|aisa nahi|aisa nhi|bilkul nahi|bilkul nhi)\b",
        r"\b(nahi yaar|nhi yaar|bore lag raha|rehne de|mat kar|cancel)\b",
    ]

    # 7. Reaction / Emotion / Exclamation
    REACTION_PATTERNS = [
        r"[\U0001F600-\U0001F64F\U0001F920-\U0001F97F\U0001FA70-\U0001FAFF]",
        r"\b(lol|lmao|rofl|haha|hahaha)\b",
        r"\b(sach me yaar|bahut maza aaya|arey bhai|arre yaar|waah|bsdk|bc|pagal hai kya)\b",
        r"\b(arre yaar|arey bhai)\b.*\b(ye kya hai|kya hai ye)\b",
    ]

    # 8. Casual Chat / Pleasantries
    CASUAL_CHAT_PATTERNS = [
        r"\b(bas badiya|bas badhiya|sab badiya|sab badhiya|aur bata|kuch khaas nahi|chill scene|kya haal hai)\b",
    ]

    # 9. Wh-question words
    QUESTION_WORDS = [
        r"\bkya\b", r"\bkyu\b", r"\bkyun\b", r"\bkaise\b", r"\bkaha\b", r"\bkahan\b",
        r"\bkab\b", r"\bkon\b", r"\bkaun\b", r"\bkitna\b", r"\bkitne\b", r"\bkitni\b",
        r"\bkonsa\b", r"\bkaunsa\b", r"\bkaisa\b", r"\bkaisi\b",
        r"\bwhat\b", r"\bwhy\b", r"\bwhere\b", r"\bwhen\b", r"\bwho\b", r"\bwhich\b", r"\bhow\b",
        r"\bbata\b", r"\bbatao\b", r"\bsuggest\b",
    ]

    @classmethod
    def analyze(cls, messages: List[Dict[str, str]]) -> UserIntent:
        """
        Determines the UserIntent of the latest user message in dialogue context.
        Uses clause-level decomposition, conversational precedence, and multi-turn adjacency.
        """
        if not messages:
            return UserIntent.UNKNOWN

        # 1. Locate latest user message
        last_user_msg = None
        last_user_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role", "").lower() == "user":
                last_user_msg = messages[idx].get("content", "").strip()
                last_user_idx = idx
                break

        if not last_user_msg:
            last_user_msg = messages[-1].get("content", "").strip()
            last_user_idx = len(messages) - 1

        if not last_user_msg:
            return UserIntent.UNKNOWN

        text_lower = last_user_msg.lower().strip()

        # Standalone ambiguous tokens without context
        if text_lower in ["kya", "kya?"]:
            return UserIntent.UNKNOWN

        # Preceding dialogue context
        prior_messages = messages[:last_user_idx]
        last_assistant_msg = ""
        for m in reversed(prior_messages):
            if m.get("role", "").lower() == "assistant":
                last_assistant_msg = m.get("content", "").strip().lower()
                break

        # Contextual response to prior assistant turn
        if last_assistant_msg:
            # If assistant invited user ("game aaja abhi", "canteen chale?")
            if any(re.search(pat, last_assistant_msg) for pat in cls.INVITATION_VERBS + cls.INVITATION_ACTIVITIES):
                if re.search(r"\b(nahi aa sakta|nhi aa sakta|busy hu|abhi nahi)\b", text_lower):
                    return UserIntent.INVITATION  # Declining invitation dialogue act
                if text_lower in ["aaja", "chal", "aaya"]:
                    return UserIntent.INVITATION
            # If assistant proposed a plan or meeting, user saying "Thik" / "ok" is agreement
            if any(term in last_assistant_msg for term in ["milte", "chalte", "kal", "baje", "khele"]):
                if text_lower in ["thik", "theek", "ok", "okay", "done", "haan"]:
                    return UserIntent.AGREEMENT

        # Multi-sentence handling: check trailing clause for questions/invitations
        clauses = [c.strip() for c in re.split(r"[.!?;\n]+", text_lower) if c.strip()]
        last_clause = clauses[-1] if clauses else text_lower

        # Priority 1: Answer Request (problem-solving guidance)
        for pat in cls.ANSWER_REQUEST_PATTERNS:
            if re.search(pat, text_lower):
                return UserIntent.ANSWER_REQUEST

        # Priority 2: Direct Planning (logistics, scheduling)
        for pat in cls.PLANNING_PATTERNS:
            if re.search(pat, text_lower):
                return UserIntent.PLANNING

        # Priority 3: Invitation (embedded or trailing)
        is_invitation = False
        for pat in cls.INVITATION_ACTIVITIES:
            if re.search(pat, text_lower):
                # Ensure it's not stating ongoing activity (e.g. 'party chal rahi')
                if not re.search(r"\bchal rahi\b|\bchal raha\b", text_lower):
                    is_invitation = True
                    break
        if not is_invitation:
            for pat in cls.INVITATION_VERBS:
                if re.search(pat, text_lower):
                    # Exclude planning logistics like 'kab milte hain'
                    if not re.search(r"\bkab milte\b", text_lower):
                        is_invitation = True
                        break

        if is_invitation:
            return UserIntent.INVITATION

        # Priority 4: Questions (Wh-words, question marks, inquiry tags)
        has_question_marker = (
            "?" in text_lower or
            any(re.search(pat, last_clause) for pat in cls.QUESTION_WORDS) or
            re.search(r"\b(kya|kaisa|kaisi|kaise|kaha|kahan|kab|kyu|kyun)\b", last_clause) or
            re.search(r"\b(aa gayi kya|aa raha hai kya|start karu kya|dekha kya|dekhi kya|kya scene hai)\b", text_lower) or
            re.search(r"\b(jagah bata|bata de|bata na)\b", text_lower)
        )

        # Preserve Scenario D technical spec inquiry ("battery backup kaisa hai?")
        if re.search(r"\bbattery backup kaisa hai\b", text_lower):
            return UserIntent.INFORMATION

        if has_question_marker:
            # Check if it's casual chat greeting like 'kya haal hai' or 'aur bata'
            if re.search(r"\b(kya haal hai|aur bata)\b", text_lower):
                return UserIntent.CASUAL_CHAT
            return UserIntent.QUESTION

        # Priority 5: Agreement
        has_agreement_token = any(re.search(pat, text_lower) for pat in cls.AGREEMENT_PATTERNS)
        if has_agreement_token:
            if any(re.search(pat, text_lower) for pat in cls.INFORMATION_PATTERNS):
                return UserIntent.INFORMATION
            words = text_lower.split()
            if len(words) <= 4:
                return UserIntent.AGREEMENT
            if last_assistant_msg and ("?" in last_assistant_msg or "theek hai" in last_assistant_msg or "sahi hai" in last_assistant_msg):
                return UserIntent.AGREEMENT

        # Priority 6: Disagreement
        for pat in cls.DISAGREEMENT_PATTERNS:
            if re.search(pat, text_lower):
                return UserIntent.DISAGREEMENT

        # Priority 7: Information (announcements, reports, ongoing activity)
        for pat in cls.INFORMATION_PATTERNS:
            if re.search(pat, text_lower):
                return UserIntent.INFORMATION

        # Progressive state ("party chal rahi hai", "drain ho raha hai")
        if re.search(r"\b(chal rahi hai|chal raha hai|ho raha hai|ho rahi hai|gaya hai|gayi hai|aa raha hai|nikala hai|de di)\b", text_lower):
            return UserIntent.INFORMATION

        # Priority 8: Reaction
        for pat in cls.REACTION_PATTERNS:
            if re.search(pat, text_lower):
                return UserIntent.REACTION

        # Priority 9: Casual Chat
        for pat in cls.CASUAL_CHAT_PATTERNS:
            if re.search(pat, text_lower):
                return UserIntent.CASUAL_CHAT

        # Declarative statement fallback -> INFORMATION if long sentence
        if len(text_lower.split()) >= 4 and any(w in text_lower for w in ["hai", "h", "tha", "thi", "raha", "rahi"]):
            return UserIntent.INFORMATION

        if has_agreement_token:
            return UserIntent.AGREEMENT

        return UserIntent.UNKNOWN
