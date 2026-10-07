package com.personaengine.app.future;

import java.util.List;
import java.util.UUID;

/**
 * Future extension contract for the Topic & Context Engine.
 * Responsible for tracking conversation state, active topics, topic transitions,
 * and context depth.
 */
public interface TopicEngine {

    String detectTopic(String messageContent);

    List<String> getRecentTopics(UUID conversationId);
}
