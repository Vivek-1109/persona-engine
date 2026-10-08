"""
Prompt builder for persona inference.
Converts multi-turn conversation messages into the native ChatML format
expected by Qwen2.5-Instruct.
"""

from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = "You are reproducing the communication style of the persona."
ALLOWED_ROLES = {"system", "user", "assistant"}


class PromptBuilder:
    """Centralized prompt construction and message validation."""

    @staticmethod
    def validate_messages(messages: List[Dict[str, str]]) -> None:
        """Validate conversation message structure and content."""
        if not messages or not isinstance(messages, list):
            raise ValueError("messages must be a non-empty list of message objects")

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message at index {idx} must be a dictionary")

            role = msg.get("role")
            if not role or role not in ALLOWED_ROLES:
                raise ValueError(
                    f"Message at index {idx} has invalid role '{role}'. "
                    f"Allowed roles: {sorted(list(ALLOWED_ROLES))}"
                )

            content = msg.get("content")
            if content is None or not isinstance(content, str) or len(content.strip()) == 0:
                raise ValueError(f"Message at index {idx} has empty or non-string content")

    @classmethod
    def prepare_messages(
        cls,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Validates messages and ensures a system turn is present at index 0.
        """
        cls.validate_messages(messages)

        cleaned = []
        for m in messages:
            cleaned.append({
                "role": m["role"].strip().lower(),
                "content": m["content"].strip()
            })

        # Inject system prompt if not present at start
        if not cleaned or cleaned[0]["role"] != "system":
            sys_text = system_prompt or SYSTEM_PROMPT
            cleaned.insert(0, {"role": "system", "content": sys_text})

        return cleaned

    @classmethod
    def build_prompt_text(
        cls,
        messages: List[Dict[str, str]],
        tokenizer: Optional[Any] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Converts messages into prompt string with generation prompt appended.
        Uses tokenizer.apply_chat_template if available; otherwise uses native ChatML format.
        """
        prepared = cls.prepare_messages(messages, system_prompt=system_prompt)

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                prepared,
                tokenize=False,
                add_generation_prompt=True
            )

        # Native Qwen ChatML fallback representation
        formatted = []
        for turn in prepared:
            formatted.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>")
        formatted.append("<|im_start|>assistant\n")
        return "\n".join(formatted)
