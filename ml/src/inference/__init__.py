"""
Inference module for Stage 5 Persona Engine.
Exports PersonaGenerator, ModelLoader, GenerationConfig, and PromptBuilder.
"""

from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator
from ml.src.inference.prompt_builder import PromptBuilder

# Compatibility alias for earlier foundation tests
GenerationParams = GenerationConfig

__all__ = [
    "GenerationConfig",
    "GenerationParams",
    "ModelLoader",
    "PersonaGenerator",
    "PromptBuilder",
]
