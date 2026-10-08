#!/usr/bin/env python3
"""
Persona Engine — Persona Candidate Extractor
Extracts instruction-tuned training candidates from conversation sessions.
Generates multi-turn context windows (1, 2, 4, 6 turns), derives behavioral annotations,
applies quality flags (KEEP, REVIEW, EXCLUDE), and maintains clean metadata separation.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import re
import sys
from pathlib import Path

# Ensure ml root in sys.path
ml_root = Path(__file__).resolve().parents[2]
if str(ml_root) not in sys.path:
    sys.path.insert(0, str(ml_root))

from src.data.conversation_segmenter import ConversationSession, DialogueTurn
from src.data.behavioral_annotator import BehavioralAnnotator, BehavioralAnnotation


class CandidateExtractor:
    """
    Extracts multi-turn conversational candidates for persona training.
    """

    def __init__(
        self,
        system_prompt: str = "You are reproducing the communication style of the persona.",
        context_turn_sizes: Optional[List[int]] = None,
        max_response_chars_for_review: int = 500,
        persona_name: str = "Vivek",
    ):
        self.system_prompt = system_prompt
        self.context_turn_sizes = context_turn_sizes or [1, 2, 4, 6]
        self.max_response_chars_for_review = max_response_chars_for_review
        self.persona_name = persona_name
        self.annotator = BehavioralAnnotator()

    def extract_from_session(
        self, session: ConversationSession
    ) -> List[Dict[str, Any]]:
        """
        Extract candidate examples from a single conversation session.
        """
        candidates: List[Dict[str, Any]] = []
        turns = session.turns

        for turn_idx, turn in enumerate(turns):
            # Target must ALWAYS be an assistant (persona) turn
            if turn.role != "assistant":
                continue

            # Must have at least one preceding user turn
            if turn_idx == 0:
                continue

            target_text = turn.text.strip()
            if not target_text:
                continue

            # Drop WhatsApp protocol artifacts from targets
            if re.search(r"waiting for this message", target_text, re.IGNORECASE):
                continue

            # Identify target message id (first message id in the turn)
            target_msg_id = turn.message_ids[0] if turn.message_ids else -1

            # Get immediate preceding user turn for context
            preceding_user_turn = turns[turn_idx - 1]
            preceding_user_text = preceding_user_turn.text.strip()

            # Annotate target response
            annotation: BehavioralAnnotation = self.annotator.annotate(
                target_text, preceding_user_text=preceding_user_text
            )

            # Determine quality flag
            if not target_text:
                quality_flag = "EXCLUDE"
            elif annotation.contains_sensitive_pii:
                quality_flag = "REVIEW"
            elif len(target_text) > self.max_response_chars_for_review:
                quality_flag = "REVIEW"
            else:
                quality_flag = "KEEP"

            # Generate context windows for each specified depth
            for k in self.context_turn_sizes:
                # k = dialogue turns of context:
                # k=1: 1 preceding turn (user) -> 1 preceding turn
                # k=2: 3 preceding turns (user, asst, user) -> 2 full dialogue exchanges
                # k=4: 7 preceding turns
                # k=6: 11 preceding turns
                num_preceding_turns = 2 * k - 1
                start_idx = turn_idx - num_preceding_turns

                # If this turn doesn't have enough history for k, we only generate
                # the k-turn example if start_idx >= 0.
                # However, for k=1, start_idx = turn_idx - 1, which is always >= 0.
                if start_idx < 0:
                    continue

                # Slice context turns
                context_slice = turns[start_idx:turn_idx]

                # Ensure context starts with user
                if context_slice[0].role != "user":
                    continue

                # Drop if any context turn contains protocol artifacts
                if any(re.search(r"waiting for this message", ct.text, re.IGNORECASE) for ct in context_slice):
                    continue

                # Build messages array in standard ChatML / OpenAI chat format
                messages = [{"role": "system", "content": self.system_prompt}]
                for ct in context_slice:
                    messages.append({"role": "user" if ct.role == "user" else "assistant", "content": ct.text})
                # Add target
                messages.append({"role": "assistant", "content": target_text})

                candidate_id = f"{session.conversation_id}_tgt{target_msg_id}_ctx{k}"

                metadata = {
                    "example_id": candidate_id,
                    "conversation_id": session.conversation_id,
                    "target_message_id": target_msg_id,
                    "timestamp": turn.timestamp,
                    "speaker": self.persona_name,
                    "context_turns": k,
                    "language": annotation.language,
                    "tone": annotation.tone,
                    "response_type": annotation.response_type,
                    "length_category": annotation.length_category,
                    "quality_flag": quality_flag,
                    "emoji_present": annotation.emoji_present,
                    "question_present": annotation.question_present,
                    "exclamation_present": annotation.exclamation_present,
                    "contains_sensitive_pii": annotation.contains_sensitive_pii,
                    "sensitive_categories": annotation.sensitive_categories,
                }

                candidates.append({
                    "messages": messages,
                    "metadata": metadata,
                })

        return candidates

    def extract_all(
        self, sessions: List[ConversationSession]
    ) -> List[Dict[str, Any]]:
        """Extract candidate examples across all conversation sessions."""
        all_candidates: List[Dict[str, Any]] = []
        for sess in sessions:
            all_candidates.extend(self.extract_from_session(sess))
        return all_candidates
