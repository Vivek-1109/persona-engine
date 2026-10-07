package com.personaengine.app.orchestrator;

import com.personaengine.app.gateway.ModelGateway;
import com.personaengine.app.gateway.ModelRequest;
import com.personaengine.app.gateway.ModelResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Component
public class MockConversationOrchestrator implements ConversationOrchestrator {

    private static final Logger log = LoggerFactory.getLogger(MockConversationOrchestrator.class);

    private final ModelGateway modelGateway;

    public MockConversationOrchestrator(ModelGateway modelGateway) {
        this.modelGateway = modelGateway;
    }

    @Override
    public OrchestrationResult orchestrateResponse(OrchestrationContext context) {
        log.info("Conversation orchestrator invoked for conversation: [{}] and persona: [{}]",
                context.getConversation().getId(),
                context.getPersona().getName());

        List<String> historyStrings = context.getHistory() != null
                ? context.getHistory().stream()
                .map(m -> m.getSenderType() + ": " + m.getContent())
                .collect(Collectors.toList())
                : new ArrayList<>();

        ModelRequest modelRequest = ModelRequest.builder()
                .conversationId(context.getConversation().getId())
                .personaId(context.getPersona().getId())
                .personaName(context.getPersona().getName())
                .personaDescription(context.getPersona().getDescription())
                .userMessage(context.getUserMessage().getContent())
                .recentHistory(historyStrings)
                .build();

        ModelResponse modelResponse = modelGateway.generateResponse(modelRequest);

        // Build rich metadata for future ML compatibility (Section 9)
        Map<String, Object> metadata = new HashMap<>(modelResponse.getMetadata());
        metadata.put("orchestrator", "MockConversationOrchestrator");
        metadata.put("topic", "general_conversation");
        metadata.put("emotion", "neutral");
        metadata.put("intent", "chat_message");
        metadata.put("conversationStrategy", "DIRECT_ANSWER");
        metadata.put("memoryReferences", List.of());

        log.info("Orchestration completed successfully for conversation: [{}]",
                context.getConversation().getId());

        return OrchestrationResult.builder()
                .content(modelResponse.getContent())
                .metadata(metadata)
                .build();
    }
}
