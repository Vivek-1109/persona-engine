package com.personaengine.app.gateway;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
public class MockModelGateway implements ModelGateway {

    private static final Logger log = LoggerFactory.getLogger(MockModelGateway.class);

    @Override
    public ModelResponse generateResponse(ModelRequest request) {
        log.info("Model gateway invoked for persona: [{}] (id: {})", request.getPersonaName(), request.getPersonaId());

        String personaName = request.getPersonaName() != null ? request.getPersonaName() : "Persona";
        String userMsg = request.getUserMessage();

        String responseText;
        if (userMsg.toLowerCase().contains("hello") || userMsg.toLowerCase().contains("hi")) {
            responseText = String.format("Hey there! %s here. Great to connect with you!", personaName);
        } else if (userMsg.toLowerCase().contains("who are you") || userMsg.toLowerCase().contains("what are you")) {
            String desc = request.getPersonaDescription() != null ? request.getPersonaDescription() : "a digital persona.";
            responseText = String.format("I'm %s. %s", personaName, desc);
        } else if (userMsg.toLowerCase().contains("how are you")) {
            responseText = String.format("Doing well! Ready to chat whenever you are. What's on your mind?");
        } else {
            responseText = String.format("Got your message: \"%s\". As %s, I'm analyzing your conversation patterns and conversational context. (Phase 1 mock response)", 
                userMsg.length() > 50 ? userMsg.substring(0, 47) + "..." : userMsg,
                personaName);
        }

        Map<String, Object> metadata = new HashMap<>();
        metadata.put("modelUsed", "mock-v1");
        metadata.put("confidence", 0.95);
        metadata.put("orchestrationStrategy", "DIRECT_MOCK");

        log.info("Generation completed successfully for persona: [{}]", personaName);

        return ModelResponse.builder()
                .content(responseText)
                .modelUsed("mock-v1")
                .metadata(metadata)
                .build();
    }
}
