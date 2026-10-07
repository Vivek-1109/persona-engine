package com.personaengine.app.gateway;

/**
 * Clean abstraction for model inference.
 * The core backend communicates with this interface rather than
 * directly binding to any specific model provider (OpenAI, vLLM, LlamaCpp, etc.).
 */
public interface ModelGateway {
    ModelResponse generateResponse(ModelRequest request);
}
