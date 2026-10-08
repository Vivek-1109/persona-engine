"""
FastAPI dependency injection providers for Persona Inference Service.
"""

from typing import Generator
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator

_generator_instance: PersonaGenerator = PersonaGenerator(model_id="stage5_v1")


def get_persona_generator() -> PersonaGenerator:
    """Dependency provider returning singleton PersonaGenerator."""
    return _generator_instance


def get_model_loader() -> type[ModelLoader]:
    """Dependency provider returning ModelLoader class."""
    return ModelLoader
