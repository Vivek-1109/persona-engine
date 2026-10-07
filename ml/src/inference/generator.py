"""
Persona Engine — Persona Generator Interface
Provides generation abstraction for loading base models, attaching fine-tuned PEFT/LoRA adapters,
and generating persona-aligned responses with streaming and sampling controls.
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

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=lambda: ["<|im_end|>", "\nUser:"])


class PersonaGenerator:
    """
    Inference generator for persona-aligned conversational generation.
    Loads quantized open-weight base models, attaches trained LoRA adapters,
    and runs sampling for interactive chat.
    """

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        adapter_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.base_model_name = base_model_name_or_path or "Qwen/Qwen2.5-1.5B-Instruct"
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.device = device

        self.is_model_loaded: bool = False
        self.is_adapter_loaded: bool = False
        self._model = None
        self._tokenizer = None

    def load_model(self, model_name: Optional[str] = None, mock: bool = False) -> None:
        """
        Loads base foundation model and tokenizer.
        If mock is True, simulates load without downloading model weights.
        """
        target = model_name or self.base_model_name

        if mock:
            logger.info(f"PersonaGenerator: Initialized mock handle for '{target}'")
            self.is_model_loaded = True
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"PersonaGenerator: Loading tokenizer for '{target}'")
            self._tokenizer = AutoTokenizer.from_pretrained(
                target,
                trust_remote_code=True,
                padding_side="left",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            has_cuda = torch.cuda.is_available()

            if has_cuda:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    target,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    target,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )

            self.is_model_loaded = True
            logger.info(f"PersonaGenerator: Base model '{target}' loaded successfully.")

            if self.adapter_path and self.adapter_path.exists():
                self.load_adapter(self.adapter_path)

        except Exception as e:
            logger.warning(f"Could not load real weights ({e}). Falling back to placeholder handle.")
            self.is_model_loaded = True

    def load_adapter(self, adapter_path: Union[str, Path]) -> None:
        """
        Loads and attaches trained LoRA adapter weights on top of the base model.
        """
        path = Path(adapter_path)
        if not path.exists():
            logger.warning(f"Adapter path not found: {path}")
            return

        try:
            from peft import PeftModel

            if self._model is not None:
                logger.info(f"PersonaGenerator: Attaching LoRA adapter from '{path}'")
                self._model = PeftModel.from_pretrained(self._model, str(path))
            self.adapter_path = path
            self.is_adapter_loaded = True
        except Exception as e:
            logger.error(f"Failed to attach LoRA adapter: {e}")

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
        system_prompt: Optional[str] = None,
        memory_context: Optional[List[str]] = None,
        persona_profile: Optional[Dict[str, Any]] = None,
        params: Optional[GenerationParams] = None,
    ) -> str:
        """
        Generates a persona-aligned response given conversation history.
        Uses the loaded model and fine-tuned LoRA weights if present;
        falls back to placeholder if weights are not loaded.
        """
        gen_params = params or GenerationParams()

        if self._model is None or self._tokenizer is None:
            # Deterministic placeholder fallback
            return f"[Persona Model Response Placeholder: Context received ({len(messages)} turns)]"

        import torch

        # Build message turns
        formatted_messages = []
        sys_content = system_prompt or (
            "You are Vivek. Respond in your learned natural conversational style, "
            "humor, and casual Hinglish messaging as you do with friends."
        )
        formatted_messages.append({"role": "system", "content": sys_content})

        for m in messages:
            role = m.get("role") or m.get("speaker") or "user"
            content = m.get("content") or m.get("text") or ""
            role_mapped = "assistant" if str(role).lower() in {"vivek", "persona", "assistant"} else "user"
            formatted_messages.append({"role": role_mapped, "content": str(content)})

        # Format prompt
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                prompt_text = self._tokenizer.apply_chat_template(
                    formatted_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in formatted_messages]
                parts.append("<|im_start|>assistant\n")
                prompt_text = "".join(parts)
        else:
            parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in formatted_messages]
            parts.append("<|im_start|>assistant\n")
            prompt_text = "".join(parts)

        device = self._model.device if hasattr(self._model, "device") else "cuda"
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=gen_params.max_new_tokens,
                temperature=gen_params.temperature,
                top_p=gen_params.top_p,
                top_k=gen_params.top_k,
                do_sample=gen_params.do_sample,
                repetition_penalty=gen_params.repetition_penalty,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][prompt_len:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
