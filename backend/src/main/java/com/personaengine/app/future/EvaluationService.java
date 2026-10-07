package com.personaengine.app.future;

import java.util.Map;
import java.util.UUID;

/**
 * Future extension contract for model evaluation, benchmark scoring,
 * and conversational alignment checks.
 */
public interface EvaluationService {

    Map<String, Object> evaluatePersonaModel(UUID personaId, String modelIdentifier);
}
