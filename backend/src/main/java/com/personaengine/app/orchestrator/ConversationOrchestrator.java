package com.personaengine.app.orchestrator;

/**
 * Conversation Orchestrator abstraction.
 * Responsible for coordinating:
 * 1. Conversation state
 * 2. Memory retrieval
 * 3. Personality state
 * 4. Topic/context analysis
 * 5. Response strategy
 * 6. Model Gateway invocation
 * 7. Response post-processing & validation
 */
public interface ConversationOrchestrator {

    OrchestrationResult orchestrateResponse(OrchestrationContext context);
}
