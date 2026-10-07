"""
Persona Engine — Annotation Schema & Models
Handles conversational annotation data structures and validation.
Supports optional behavioral signals: emotion, tone, humor, sarcasm, teasing,
language mixing, response strategy, and relationship context.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import jsonschema
except ImportError:
    jsonschema = None


@dataclass
class MessageAnnotation:
    """Represents fine-grained behavioral signals for a single message turn."""

    topic: Optional[str] = None
    intent: Optional[str] = None  # question, statement, joke, request, vent, tease
    emotion: Optional[str] = None  # happy, sad, frustrated, neutral, excited, anxious
    tone: Optional[str] = None  # formal, casual, sarcastic, playful, serious
    humor: Optional[bool] = None
    sarcasm: Optional[bool] = None
    teasing: Optional[bool] = None
    language: Optional[str] = None  # en, hi, hinglish
    language_mixing: Optional[bool] = None
    response_strategy: Optional[str] = None  # DIRECT_ANSWER, FOLLOW_UP, TEASE, JOKE, SUPPORT
    relationship_context: Optional[str] = None  # close_friend, acquaintance, formal

    def to_dict(self) -> Dict[str, Any]:
        """Returns non-null attributes as a dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class AnnotationValidator:
    """Validates annotated conversations against annotation.schema.json."""

    def __init__(self, schema_path: Optional[Path] = None):
        if schema_path and schema_path.is_file():
            with open(schema_path, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        else:
            default_path = (
                Path(__file__).resolve().parents[2]
                / "schemas"
                / "annotation.schema.json"
            )
            if default_path.is_file():
                with open(default_path, "r", encoding="utf-8") as f:
                    self.schema = json.load(f)
            else:
                self.schema = None

    def validate(self, annotated_conv: Dict[str, Any]) -> bool:
        if self.schema:
            jsonschema.validate(instance=annotated_conv, schema=self.schema)
        return True


def create_annotated_conversation(
    conversation_id: str,
    messages: List[Dict[str, Any]],
    persona_id: Optional[str] = "default",
    source: Optional[str] = "manual",
) -> Dict[str, Any]:
    """Helper to structure an annotated conversation."""
    return {
        "conversation_id": conversation_id,
        "source": source,
        "persona_id": persona_id,
        "messages": messages,
    }


def validate_annotated_conversation(conv: Dict[str, Any]) -> bool:
    """Convenience validator for an annotated conversation."""
    validator = AnnotationValidator()
    return validator.validate(conv)
