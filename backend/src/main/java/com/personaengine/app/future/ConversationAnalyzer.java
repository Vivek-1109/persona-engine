package com.personaengine.app.future;

import java.util.Map;

/**
 * Future extension contract for analyzing conversation signals:
 * intent, emotion, sentiment, sarcasm, teasing, question behavior.
 */
public interface ConversationAnalyzer {

    Map<String, Object> analyze(String messageContent);
}
