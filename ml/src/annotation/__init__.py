"""
Persona Engine — Annotation Subpackage
"""

from .schema import (
    AnnotationValidator,
    MessageAnnotation,
    create_annotated_conversation,
    validate_annotated_conversation,
)

__all__ = [
    "MessageAnnotation",
    "AnnotationValidator",
    "create_annotated_conversation",
    "validate_annotated_conversation",
]
