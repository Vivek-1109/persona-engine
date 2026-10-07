package com.personaengine.app.future;

import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Future extension contract for the Memory Engine.
 * Responsible for short-term, long-term, episodic, and semantic memory.
 */
public interface MemoryEngine {

    List<Map<String, Object>> retrieveMemories(UUID personaId, UUID conversationId, String query);

    void storeMemory(UUID personaId, UUID conversationId, Map<String, Object> memoryItem);
}
