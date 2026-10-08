#!/usr/bin/env python3
"""
Persona Engine — Conversation Segmenter
Groups cleaned WhatsApp messages into distinct, chronological conversation sessions
based on a configurable inactivity threshold (CONVERSATION_GAP_MINUTES).
Merges consecutive same-speaker bubbles into coherent dialogue turns.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

# Ensure ml root in sys.path
ml_root = Path(__file__).resolve().parents[2]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.data.whatsapp_parser import ParsedMessage


@dataclass
class DialogueTurn:
    """Represents a coherent dialogue turn by a single speaker."""
    speaker: str  # "Vivek Jha😎" or partner
    role: str  # "assistant" or "user"
    text: str  # Full merged text for this turn
    timestamp: str  # Timestamp of the last bubble in this turn
    start_timestamp: str  # Timestamp of the first bubble in this turn
    message_ids: List[int]  # Line numbers / message IDs included in this turn


@dataclass
class ConversationSession:
    """Represents a discrete chronological conversation episode."""
    conversation_id: str
    start_time: str
    end_time: str
    duration_minutes: float
    turns: List[DialogueTurn]
    raw_message_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to serializable dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": round(self.duration_minutes, 2),
            "raw_message_count": self.raw_message_count,
            "turns": [asdict(t) for t in self.turns],
        }


class ConversationSegmenter:
    """
    Segments WhatsApp messages into conversation episodes based on inactivity gap.
    """

    def __init__(
        self,
        persona_speaker: str = "Vivek Jha😎",
        gap_minutes: int = 120,
        min_messages: int = 2,
    ):
        self.persona_speaker = persona_speaker
        self.gap_minutes = gap_minutes
        self.min_messages = min_messages

    def segment(self, messages: List[ParsedMessage]) -> List[ConversationSession]:
        """
        Segment a list of parsed messages into conversation sessions.
        Filters out non-conversational types (system, media, deleted, empty, pure url).
        """
        # Filter to usable text messages
        usable = [
            m for m in messages
            if m.message_type == "text" and m.text.strip()
        ]

        if not usable:
            return []

        # Group messages by inactivity gap threshold
        raw_sessions: List[List[ParsedMessage]] = []
        current_session: List[ParsedMessage] = []

        for msg in usable:
            if not current_session:
                current_session.append(msg)
                continue

            try:
                prev_dt = datetime.strptime(current_session[-1].timestamp, "%Y-%m-%d %H:%M:%S")
                curr_dt = datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M:%S")
                delta_minutes = (curr_dt - prev_dt).total_seconds() / 60.0
            except ValueError:
                delta_minutes = 0.0

            if delta_minutes > self.gap_minutes:
                raw_sessions.append(current_session)
                current_session = [msg]
            else:
                current_session.append(msg)

        if current_session:
            raw_sessions.append(current_session)

        # Build ConversationSession objects
        sessions: List[ConversationSession] = []
        for idx, raw_msgs in enumerate(raw_sessions, 1):
            if len(raw_msgs) < self.min_messages:
                continue

            # Merge consecutive messages by the same speaker into turns
            turns: List[DialogueTurn] = []
            for m in raw_msgs:
                role = "assistant" if m.speaker == self.persona_speaker else "user"
                clean_text = m.text.strip()

                if not turns:
                    turns.append(
                        DialogueTurn(
                            speaker=m.speaker,
                            role=role,
                            text=clean_text,
                            timestamp=m.timestamp,
                            start_timestamp=m.timestamp,
                            message_ids=[m.line_number],
                        )
                    )
                elif turns[-1].speaker == m.speaker:
                    # Same speaker: append text with newline
                    turns[-1].text += "\n" + clean_text
                    turns[-1].timestamp = m.timestamp
                    turns[-1].message_ids.append(m.line_number)
                else:
                    turns.append(
                        DialogueTurn(
                            speaker=m.speaker,
                            role=role,
                            text=clean_text,
                            timestamp=m.timestamp,
                            start_timestamp=m.timestamp,
                            message_ids=[m.line_number],
                        )
                    )

            # Check if this conversation has at least 1 persona turn and 1 user turn
            has_persona = any(t.role == "assistant" for t in turns)
            has_user = any(t.role == "user" for t in turns)
            if not (has_persona and has_user):
                continue

            start_t = raw_msgs[0].timestamp
            end_t = raw_msgs[-1].timestamp
            try:
                dt_start = datetime.strptime(start_t, "%Y-%m-%d %H:%M:%S")
                dt_end = datetime.strptime(end_t, "%Y-%m-%d %H:%M:%S")
                duration = max(0.0, (dt_end - dt_start).total_seconds() / 60.0)
            except ValueError:
                duration = 0.0

            conv_id = f"conv_{idx:04d}"
            sessions.append(
                ConversationSession(
                    conversation_id=conv_id,
                    start_time=start_t,
                    end_time=end_t,
                    duration_minutes=duration,
                    turns=turns,
                    raw_message_count=len(raw_msgs),
                )
            )

        return sessions
