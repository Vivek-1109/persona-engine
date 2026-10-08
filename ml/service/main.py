"""
FastAPI Inference Service for Stage 5 Persona Engine.
Provides clean HTTP REST endpoints for persona text generation and health monitoring.
"""

from contextlib import asynccontextmanager
import logging
import uuid
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ml.service.dependencies import get_model_loader, get_persona_generator
from ml.service.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ReadyResponse,
)
from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.model_loader import ModelLoader
from ml.src.inference.persona_generator import PersonaGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PersonaService")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for application startup and shutdown."""
    logger.info("Initializing Persona Inference Service (Stage 5 baseline)...")
    info = ModelLoader.get_info()
    logger.info(f"Model status on startup: loaded={info['loaded']}, model_id={info['model_id']}")
    yield
    logger.info("Shutting down Persona Inference Service.")


app = FastAPI(
    title="Persona Engine Inference API",
    description="Standalone model inference service serving the Stage 5 Vivek persona baseline.",
    version="5.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,  # Keep minimal
)


# =====================================================================
# Error Handlers (Clean responses, no internal traces exposed)
# =====================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Clean HTTP 422/400 validation error formatting without leaking stack traces."""
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append(f"{loc}: {msg}")
    logger.warning(f"Validation failure on {request.url.path}: {errors}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid request payload", "details": errors}
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handles domain validation errors as clean HTTP 400."""
    logger.warning(f"Bad request value on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Bad request", "details": [str(exc)]}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches unhandled server errors, logs detailed server-side, returns clean 500."""
    req_id = str(uuid.uuid4())[:8]
    logger.error(f"[ReqID={req_id}] Internal server error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error occurred during inference", "request_id": req_id}
    )


# =====================================================================
# API Endpoints
# =====================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["Diagnostics"]
)
async def health_check() -> HealthResponse:
    """Returns service process vitality status."""
    return HealthResponse(status="ok", model="stage5_v1")


@app.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Model Readiness Check",
    tags=["Diagnostics"]
)
async def readiness_check(
    loader: type[ModelLoader] = Depends(get_model_loader)
) -> ReadyResponse:
    """Returns whether model weights and tokenizer are loaded in memory."""
    is_ready = loader.is_loaded()
    return ReadyResponse(ready=is_ready, model="stage5_v1")


@app.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Generate Persona Response",
    tags=["Inference"]
)
async def generate_response(
    payload: GenerateRequest,
    generator: PersonaGenerator = Depends(get_persona_generator)
) -> GenerateResponse:
    """
    Generates a conversational persona response from multi-turn input messages.
    """
    req_id = str(uuid.uuid4())[:8]

    # Convert pydantic ChatMessage list to dict list
    messages_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in payload.messages
    ]

    # Convert optional generation settings to GenerationConfig
    gen_config = None
    if payload.generation is not None:
        gen_config = GenerationConfig.from_dict(payload.generation.model_dump())

    try:
        response_text = generator.generate(
            messages=messages_dicts,
            generation_config=gen_config,
            request_id=req_id
        )
        return GenerateResponse(response=response_text, model=generator.model_id)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"[ReqID={req_id}] Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model generation failed"
        )
