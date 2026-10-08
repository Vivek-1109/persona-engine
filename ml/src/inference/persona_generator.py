"""
Pure Persona Generator component.
Executes generation using loaded Qwen2.5-1.5B model + Stage 5 adapter.
Does not know about databases, Spring Boot, Redis, or external persistence.
Supports both modern ChatML format and backwards-compatible message schemas.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.prompt_builder import PromptBuilder

logger = logging.getLogger("PersonaGenerator")


class PersonaGenerator:
    """Core persona text generation interface."""

    def __init__(
        self,
        model_id: str = "stage5_v1",
        base_model_name_or_path: Optional[str] = None,
        adapter_path: Optional[Union[str, Any]] = None,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.base_model_name = base_model_name_or_path or "Qwen/Qwen2.5-1.5B-Instruct"
        self._mock_mode = False

    @property
    def is_model_loaded(self) -> bool:
        """Returns True if the underlying model is loaded in memory."""
        return ModelLoader.is_loaded()

    def load_model(self, model_name: Optional[str] = None, mock: bool = False) -> None:
        """Loads model into memory. Supports mock=True for lightweight testing."""
        if mock:
            from unittest.mock import MagicMock
            import torch

            mock_model = MagicMock()
            mock_model.parameters.side_effect = lambda: iter([torch.zeros(1)])
            mock_model.generate.return_value = torch.tensor([[10, 20, 30, 40, 50]])
            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "Persona Model Response Placeholder: kuch nahi bhai chill"

            self._mock_mode = True
            ModelLoader.load_model(mock_model=mock_model, mock_tokenizer=mock_tokenizer)
        else:
            self._mock_mode = False
            ModelLoader.load_model(model_id=self.model_id)

    def unload_model(self) -> None:
        """Unloads model from memory."""
        self._mock_mode = False
        ModelLoader.unload()

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Normalize messages supporting both role/content and speaker/text formats."""
        normalized = []
        for m in messages:
            if "role" in m and "content" in m:
                normalized.append({
                    "role": str(m["role"]).strip().lower(),
                    "content": str(m["content"]).strip()
                })
            elif "speaker" in m and "text" in m:
                spk = str(m["speaker"]).strip().lower()
                role = "assistant" if spk in {"persona", "vivek", "assistant"} else "user"
                normalized.append({
                    "role": role,
                    "content": str(m["text"]).strip()
                })
            else:
                # Let PromptBuilder validation handle the error
                normalized.append(m)
        return normalized

    def generate(
        self,
        messages: List[Dict[str, Any]],
        generation_config: Optional[GenerationConfig] = None,
        request_id: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Generates a persona response from conversational messages.

        Input:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
                      (or backwards-compatible {"speaker": "...", "text": "..."})
            generation_config: Optional GenerationConfig instance
            request_id: Optional correlation tracking identifier

        Output:
            Plain generated response string.
        """
        req_id = request_id or str(uuid.uuid4())[:8]
        gen_cfg = generation_config or GenerationConfig()

        # Normalize incoming format
        norm_messages = self._normalize_messages(messages)

        # Validate normalized messages
        PromptBuilder.validate_messages(norm_messages)

        # If mock mode active from load_model(mock=True), return placeholder
        if self._mock_mode:
            return "Persona Model Response Placeholder: kuch nahi bhai chill"

        # 2. Retrieve model & tokenizer (singleton)
        model, tokenizer = ModelLoader.load_model(model_id=self.model_id)

        # 3. Format input prompt
        prompt_text = PromptBuilder.build_prompt_text(norm_messages, tokenizer=tokenizer)

        # 4. Tokenize
        import torch

        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = "cpu"

        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        input_token_count = input_ids.shape[1]

        # 5. Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": gen_cfg.max_new_tokens,
            "repetition_penalty": gen_cfg.repetition_penalty,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": gen_cfg.do_sample,
        }
        if gen_cfg.do_sample:
            gen_kwargs["temperature"] = gen_cfg.temperature
            gen_kwargs["top_p"] = gen_cfg.top_p

        # 6. Execute inference & measure latency
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
        duration = time.time() - start_time

        # 7. Decode response tokens (excluding prompt)
        gen_tokens = outputs[0][input_token_count:]
        output_token_count = len(gen_tokens)
        response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # 8. Performance logging (Privacy-safe: no raw conversation text dumped)
        info = ModelLoader.get_info()
        logger.info(
            f"PersonaGenerator [ReqID={req_id}]: "
            f"Model={self.model_id} | "
            f"Device={info['device']} | "
            f"InputTurns={len(messages)} | "
            f"InputTokens={input_token_count} | "
            f"OutputTokens={output_token_count} | "
            f"Latency={duration*1000:.1f}ms"
        )

        return response_text
