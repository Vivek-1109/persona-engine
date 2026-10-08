"""
Generation configuration for Stage 5 persona inference.
Centralizes all sampling and decoding parameters.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class GenerationConfig:
    """Configurable generation parameters with defaults matching Stage 5 evaluation."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 64
    repetition_penalty: float = 1.05
    do_sample: bool = True

    def __post_init__(self):
        if self.temperature <= 0.0:
            self.temperature = 0.0
            self.do_sample = False
        else:
            self.do_sample = True

        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")

        if self.max_new_tokens < 1 or self.max_new_tokens > 2048:
            raise ValueError(f"max_new_tokens must be between 1 and 2048, got {self.max_new_tokens}")

        if self.repetition_penalty < 0.1 or self.repetition_penalty > 5.0:
            raise ValueError(f"repetition_penalty must be between 0.1 and 5.0, got {self.repetition_penalty}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GenerationConfig":
        if not data:
            return cls()
        valid_keys = {"temperature", "top_p", "max_new_tokens", "repetition_penalty"}
        filtered = {k: v for k, v in data.items() if k in valid_keys and v is not None}
        return cls(**filtered)
