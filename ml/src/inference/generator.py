"""
Persona Engine — Persona Generator Interface
Provides an extensible generation abstraction for loading base models,
attaching fine-tuned PEFT/LoRA adapters, injecting memory/context, and generating responses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    """Hyperparameters governing text decoding/sampling."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=lambda: ["<|im_end|>", "\nUser:"])


class PersonaGenerator:
    """
    Model-agnostic inference generator for persona-aligned conversational generation.
    Supports local fallback/mock execution as well as future Hugging Face / vLLM serving.
    """

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        adapter_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ):
        self.base_model_name = base_model_name_or_path or "unsloth/Qwen2.5-1.5B"
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device

        self.is_model_loaded: bool = False
        self.is_adapter_loaded: bool = False
        self._model = None
        self._tokenizer = None

    def load_model(self, model_name: Optional[str] = None) -> None:
        """
        Loads base foundation model.
        In this foundation phase, marks placeholder readiness without triggering large downloads.
        """
        target = model_name or self.base_model_name
        logger.info(f"PersonaGenerator: Initialized placeholder handle for base model '{target}' on {self.device}")
        self.is_model_loaded = True

    def load_adapter(self, adapter_path: Union[str, Path]) -> None:
        """
        Loads LoRA adapter weights on top of base model.
        """
        path = Path(adapter_path)
        logger.info(f"PersonaGenerator: Initialized adapter handle from '{path}'")
        self.adapter_path = path
        self.is_adapter_loaded = True

    def unload_model(self) -> None:
        """Frees model and GPU/CPU memory."""
        self._model = None
        self._tokenizer = None
        self.is_model_loaded = False
        self.is_adapter_loaded = False
        logger.info("PersonaGenerator: Model and adapter memory released.")

    def generate(
        self,
        messages: List[Dict[str, str]],
        memory_context: Optional[List[str]] = None,
        persona_profile: Optional[Dict[str, Any]] = None,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """
        Generates a persona-aligned response given conversation history and optional memory.
        In the foundation phase, provides a clean deterministic mock response if weights are not loaded.
        """
        gen_params = params or GenerationParams()

        # If real model was loaded into _model, run inference here.
        # Otherwise, return safe deterministic placeholder acknowledging the persona context
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("speaker", "").lower() == "user":
                last_user_msg = m.get("text", "")
                break

        logger.debug(
            f"PersonaGenerator: Generating response for '{last_user_msg}' "
            f"(temp={gen_params.temperature}, max_tokens={gen_params.max_new_tokens})"
        )

        # Baseline fallback message when model weights are not loaded
        return f"[Persona Model Response Placeholder: Context received ({len(messages)} turns)]"
