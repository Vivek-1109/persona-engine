"""
Persona Engine — Dataset Validator
Validates conversational datasets against JSON schemas and conversational domain rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    import jsonschema
except ImportError:
    jsonschema = None

from .loader import iter_jsonl, load_jsonl, normalize_conversation


@dataclass
class ValidationIssue:
    severity: str  # "ERROR" | "WARNING"
    conversation_id: Optional[str]
    message_index: Optional[int]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "conversation_id": self.conversation_id,
            "message_index": self.message_index,
            "message": self.message,
        }


@dataclass
class ValidationResult:
    is_valid: bool
    total_conversations: int = 0
    total_messages: int = 0
    error_count: int = 0
    warning_count: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Validation {status}: {self.total_conversations} conversations, "
            f"{self.total_messages} messages | Errors: {self.error_count}, "
            f"Warnings: {self.warning_count}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_conversations": self.total_conversations,
            "total_messages": self.total_messages,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
        }


class DatasetValidator:
    """
    Validates conversation datasets against JSON Schema and domain consistency rules.
    """

    def __init__(
        self,
        schema_path: Optional[Union[str, Path]] = None,
        allowed_speakers: Optional[Set[str]] = None,
        min_messages: int = 2,
        max_messages: int = 500,
        min_message_len: int = 1,
        max_message_len: int = 5000,
    ):
        self.enforce_allowed_speakers = allowed_speakers is not None
        self.allowed_speakers = (
            set(s.lower() for s in allowed_speakers)
            if allowed_speakers
            else {"user", "persona", "system"}
        )
        self.min_messages = min_messages
        self.max_messages = max_messages
        self.min_message_len = min_message_len
        self.max_message_len = max_message_len

        self.schema: Optional[Dict[str, Any]] = None
        if schema_path:
            schema_file = Path(schema_path)
            if schema_file.is_file():
                with open(schema_file, "r", encoding="utf-8") as f:
                    self.schema = json.load(f)
        else:
            # Try to auto-locate default conversation schema
            default_schema = (
                Path(__file__).resolve().parents[2]
                / "schemas"
                / "conversation.schema.json"
            )
            if default_schema.is_file():
                with open(default_schema, "r", encoding="utf-8") as f:
                    self.schema = json.load(f)

    def validate_conversation(
        self,
        conv: Dict[str, Any],
        seen_ids: Set[str],
        issues: List[ValidationIssue],
    ) -> int:
        """Validates a single conversation dictionary. Returns message count."""
        norm_conv = normalize_conversation(conv)
        conv_id = norm_conv.get("conversation_id")

        # 1. JSON Schema check if loaded
        if self.schema:
            try:
                jsonschema.validate(instance=norm_conv, schema=self.schema)
            except jsonschema.ValidationError as err:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=str(conv_id) if conv_id else "UNKNOWN",
                        message_index=None,
                        message=f"Schema violation: {err.message} (path: {list(err.path)})",
                    )
                )

        # 2. Check conversation_id
        if not conv_id:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    conversation_id=None,
                    message_index=None,
                    message="Missing conversation_id",
                )
            )
        else:
            str_id = str(conv_id)
            if str_id in seen_ids:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=str_id,
                        message_index=None,
                        message=f"Duplicate conversation_id: '{str_id}'",
                    )
                )
            seen_ids.add(str_id)

        # 3. Check messages list
        messages = norm_conv.get("messages")
        if not isinstance(messages, list):
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    conversation_id=str(conv_id),
                    message_index=None,
                    message="Field 'messages' must be a list",
                )
            )
            return 0

        msg_count = len(messages)
        if msg_count < self.min_messages:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    conversation_id=str(conv_id),
                    message_index=None,
                    message=f"Conversation has {msg_count} messages, minimum is {self.min_messages}",
                )
            )
        elif msg_count > self.max_messages:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    conversation_id=str(conv_id),
                    message_index=None,
                    message=f"Conversation exceeds maximum recommended messages ({msg_count} > {self.max_messages})",
                )
            )

        # 4. Check each message
        has_persona_turn = False
        has_user_turn = False
        persona_name = str(norm_conv.get("persona_id") or "").lower()

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=str(conv_id),
                        message_index=idx,
                        message=f"Message {idx} is not a valid dictionary/object",
                    )
                )
                continue

            speaker = msg.get("speaker")
            text = msg.get("text")

            # Speaker check
            if not speaker:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=str(conv_id),
                        message_index=idx,
                        message=f"Message {idx} missing 'speaker' field",
                    )
                )
            elif self.enforce_allowed_speakers and str(speaker).lower() not in self.allowed_speakers:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=str(conv_id),
                        message_index=idx,
                        message=f"Unknown speaker '{speaker}' in message {idx}. Allowed: {sorted(self.allowed_speakers)}",
                    )
                )
            else:
                spk_lower = str(speaker).lower()
                if spk_lower in {"persona", "assistant"} or (persona_name and spk_lower == persona_name):
                    has_persona_turn = True
                else:
                    has_user_turn = True

            # Text check
            if text is None:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=str(conv_id),
                        message_index=idx,
                        message=f"Message {idx} has null/missing text",
                    )
                )
            elif not isinstance(text, str):
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=str(conv_id),
                        message_index=idx,
                        message=f"Message {idx} text must be a string",
                    )
                )
            else:
                stripped_len = len(text.strip())
                if stripped_len < self.min_message_len:
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            conversation_id=str(conv_id),
                            message_index=idx,
                            message=f"Message {idx} is empty or whitespace only",
                        )
                    )
                elif stripped_len > self.max_message_len:
                    issues.append(
                        ValidationIssue(
                            severity="WARNING",
                            conversation_id=str(conv_id),
                            message_index=idx,
                            message=f"Message {idx} is unusually long ({stripped_len} chars)",
                        )
                    )

        # Check speaker balance
        if not has_persona_turn and len(messages) > 1:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    conversation_id=str(conv_id),
                    message_index=None,
                    message="Conversation contains no messages from target persona",
                )
            )
        if not has_user_turn and len(messages) > 1:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    conversation_id=str(conv_id),
                    message_index=None,
                    message="Conversation contains no messages from conversational partner",
                )
            )

        return msg_count

    def validate_dataset(self, conversations: List[Dict[str, Any]]) -> ValidationResult:
        """Validates an in-memory list of conversation dictionaries."""
        seen_ids: Set[str] = set()
        issues: List[ValidationIssue] = []
        total_messages = 0

        for conv in conversations:
            msg_count = self.validate_conversation(conv, seen_ids, issues)
            total_messages += msg_count

        error_count = sum(1 for i in issues if i.severity == "ERROR")
        warning_count = sum(1 for i in issues if i.severity == "WARNING")

        return ValidationResult(
            is_valid=(error_count == 0),
            total_conversations=len(conversations),
            total_messages=total_messages,
            error_count=error_count,
            warning_count=warning_count,
            issues=issues,
        )

    def validate_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """Loads and validates a .jsonl or .json dataset file."""
        path = Path(file_path)
        if not path.is_file():
            return ValidationResult(
                is_valid=False,
                error_count=1,
                issues=[
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=None,
                        message_index=None,
                        message=f"File not found: {path}",
                    )
                ],
            )

        try:
            if path.suffix == ".jsonl":
                conversations = load_jsonl(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                conversations = data if isinstance(data, list) else [data]
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_count=1,
                issues=[
                    ValidationIssue(
                        severity="ERROR",
                        conversation_id=None,
                        message_index=None,
                        message=f"File decode error: {e}",
                    )
                ],
            )

        return self.validate_dataset(conversations)


def validate_dataset_file(
    file_path: Union[str, Path],
    schema_path: Optional[Union[str, Path]] = None,
) -> ValidationResult:
    """Convenience helper to validate a dataset file."""
    validator = DatasetValidator(schema_path=schema_path)
    return validator.validate_file(file_path)
