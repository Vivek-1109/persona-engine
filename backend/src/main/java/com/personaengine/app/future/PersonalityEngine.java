package com.personaengine.app.future;

import java.util.Map;
import java.util.UUID;

/**
 * Future extension contract for the Personality Engine.
 * Responsible for learning and representing communication traits,
 * language mixing, humor, sarcasm, tone, and behavioral characteristics.
 */
public interface PersonalityEngine {

    Map<String, Object> getPersonalityProfile(UUID personaId);

    String adaptStyle(String baseContent, UUID personaId);
}
