package com.personaengine.app.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * Foundation placeholder for the persistent memory system.
 * Designed to be extended with pgvector, episodic memory, and
 * semantic retrieval in subsequent ML phases without API redesign.
 */
@Service
public class MemoryService {

    private static final Logger log = LoggerFactory.getLogger(MemoryService.class);

    public List<Map<String, Object>> retrieveMemories(UUID personaId, UUID conversationId, String query) {
        log.debug("Memory retrieval invoked for persona: {}, conversation: {}", personaId, conversationId);
        // Placeholder foundation: returns empty list until vector search / episodic memory is plugged in
        return new ArrayList<>();
    }

    public void storeMemoryCandidate(UUID personaId, UUID conversationId, String text, double importanceScore) {
        log.debug("Storing candidate memory for persona {}: importance={}", personaId, importanceScore);
        // Will persist to future memories table and pgvector index
    }
}
