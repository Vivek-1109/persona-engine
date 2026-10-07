"""
Persona Engine — Conversation Training Example Builder
Transforms multi-turn conversation dialogues into model-ready training examples
matching training_example.schema.json.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ConversationBuilder:
    """
    Constructs prompt-response pairs / chat-formatted training examples
    from multi-turn conversations.
    """

    def __init__(
        self,
        target_speaker: str = "persona",
        user_speaker: str = "user",
        system_speaker: str = "system",
        chat_template: str = "chatml",
        max_history_turns: Optional[int] = 10,
    ):
        self.target_speaker = target_speaker.lower()
        self.user_speaker = user_speaker.lower()
        self.system_speaker = system_speaker.lower()
        self.chat_template = chat_template.lower()
        self.max_history_turns = max_history_turns

    def format_turn(self, role: str, content: str) -> str:
        """Formats a single dialogue turn using standard chat templates."""
        # Map our roles to chat template roles
        role_map = {
            self.user_speaker: "user",
            self.target_speaker: "assistant",
            self.system_speaker: "system",
        }
        mapped_role = role_map.get(role.lower(), role)

        if self.chat_template == "chatml":
            return f"<|im_start|>{mapped_role}\n{content}<|im_end|>\n"
        elif self.chat_template == "llama":
            if mapped_role == "system":
                return f"<<SYS>>\n{content}\n<</SYS>>\n\n"
            elif mapped_role == "user":
                return f"[INST] {content} [/INST]"
            else:
                return f" {content} "
        else:
            # Simple fallback format
            return f"{mapped_role.capitalize()}: {content}\n"

    def build_examples_from_conversation(
        self,
        conversation: Dict[str, Any],
        split: Optional[str] = "train",
    ) -> List[Dict[str, Any]]:
        """
        Extracts training examples from a single conversation.
        For every persona turn, generates a training example containing
        the preceding dialogue context and the persona response.
        """
        conv_id = str(conversation.get("conversation_id", "unknown"))
        persona_id = str(conversation.get("persona_id", "default"))
        messages = conversation.get("messages", [])

        if not messages or len(messages) < 2:
            return []

        examples: List[Dict[str, Any]] = []

        # Find persona turns to use as training targets
        history_buffer: List[Dict[str, str]] = []

        for msg in messages:
            speaker = str(msg.get("speaker", "")).lower()
            text = str(msg.get("text", "")).strip()

            if not text:
                continue

            if speaker == self.target_speaker and len(history_buffer) > 0:
                # Target found! Build training sequence
                turns_to_use = (
                    history_buffer[-self.max_history_turns :]
                    if self.max_history_turns
                    else history_buffer
                )

                formatted_parts: List[str] = []
                for h in turns_to_use:
                    formatted_parts.append(self.format_turn(h["speaker"], h["text"]))

                # Append current target persona response
                formatted_parts.append(self.format_turn(speaker, text))
                full_text = "".join(formatted_parts).strip()

                example: Dict[str, Any] = {
                    "conversation_id": conv_id,
                    "text": full_text,
                    "persona_id": persona_id,
                    "metadata": {
                        "num_turns": len(turns_to_use) + 1,
                        "target_speaker": self.target_speaker,
                    },
                }
                if split:
                    example["split"] = split

                examples.append(example)

            # Add to history buffer
            history_buffer.append({"speaker": speaker, "text": text})

        return examples

    def build_dataset(
        self,
        conversations: List[Dict[str, Any]],
        split: Optional[str] = "train",
    ) -> List[Dict[str, Any]]:
        """Builds all training examples from a collection of conversations."""
        dataset: List[Dict[str, Any]] = []
        for conv in conversations:
            dataset.extend(self.build_examples_from_conversation(conv, split=split))
        return dataset


def build_training_examples(
    conversations: List[Dict[str, Any]],
    split: Optional[str] = "train",
    chat_template: str = "chatml",
) -> List[Dict[str, Any]]:
    """Convenience helper to build training examples."""
    builder = ConversationBuilder(chat_template=chat_template)
    return builder.build_dataset(conversations, split=split)
