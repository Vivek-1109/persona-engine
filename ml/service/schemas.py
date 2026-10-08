"""
Pydantic schemas for Persona Inference Service.
Enforces strict input validation on conversation turns and generation parameters.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """A single turn in the conversational dialogue."""
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Role of the turn speaker. Allowed values: system, user, assistant."
    )
    content: str = Field(
        ...,
        description="Text content of the message. Cannot be empty."
    )

    @field_validator("content")
    @classmethod
    def validate_non_empty_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty or whitespace only")
        return v.strip()


class GenerationSettings(BaseModel):
    """Optional generation hyperparameters."""
    temperature: Optional[float] = Field(
        default=0.7, ge=0.0, le=2.0,
        description="Sampling temperature. 0.0 for deterministic greedy decoding."
    )
    top_p: Optional[float] = Field(
        default=0.9, ge=0.0, le=1.0,
        description="Nucleus sampling threshold."
    )
    max_new_tokens: Optional[int] = Field(
        default=64, ge=1, le=512,
        description="Maximum tokens to generate."
    )
    repetition_penalty: Optional[float] = Field(
        default=1.05, ge=0.5, le=3.0,
        description="Repetition penalty applied during generation."
    )


class GenerateRequest(BaseModel):
    """Payload for POST /generate."""
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="List of conversation messages."
    )
    generation: Optional[GenerationSettings] = Field(
        default=None,
        description="Optional generation sampling parameters."
    )


class GenerateResponse(BaseModel):
    """Response payload for POST /generate."""
    response: str = Field(..., description="Generated persona response text.")
    model: str = Field(default="stage5_v1", description="Identifier of the serving model.")


class HealthResponse(BaseModel):
    """Response payload for GET /health."""
    status: str = Field(default="ok", description="Service process vitality indicator.")
    model: str = Field(default="stage5_v1", description="Registered model ID.")


class ReadyResponse(BaseModel):
    """Response payload for GET /ready."""
    ready: bool = Field(..., description="Whether model weights are loaded and ready for inference.")
    model: str = Field(default="stage5_v1", description="Registered model ID.")
