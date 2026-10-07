package com.personaengine.app.service;

import com.personaengine.app.dto.request.SendMessageRequest;
import com.personaengine.app.dto.response.MessageResponse;
import com.personaengine.app.dto.response.SendMessageResponse;
import com.personaengine.app.entity.Conversation;
import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.entity.Message;
import com.personaengine.app.entity.Persona;
import com.personaengine.app.entity.SenderType;
import com.personaengine.app.exception.ResourceNotFoundException;
import com.personaengine.app.orchestrator.ConversationOrchestrator;
import com.personaengine.app.orchestrator.OrchestrationContext;
import com.personaengine.app.orchestrator.OrchestrationResult;
import com.personaengine.app.repository.ConversationRepository;
import com.personaengine.app.repository.MessageRepository;
import com.personaengine.app.repository.PersonaRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@Transactional
public class MessageService {

    private static final Logger log = LoggerFactory.getLogger(MessageService.class);

    private final MessageRepository messageRepository;
    private final ConversationRepository conversationRepository;
    private final PersonaRepository personaRepository;
    private final ConversationOrchestrator conversationOrchestrator;

    public MessageService(MessageRepository messageRepository,
                          ConversationRepository conversationRepository,
                          PersonaRepository personaRepository,
                          ConversationOrchestrator conversationOrchestrator) {
        this.messageRepository = messageRepository;
        this.conversationRepository = conversationRepository;
        this.personaRepository = personaRepository;
        this.conversationOrchestrator = conversationOrchestrator;
    }

    public SendMessageResponse sendMessage(UUID conversationId, SendMessageRequest request) {
        log.info("Processing message for conversation: [{}]", conversationId);

        Conversation conversation = conversationRepository.findById(conversationId)
                .orElseThrow(() -> new ResourceNotFoundException("Conversation not found with id: " + conversationId));

        if (conversation.getStatus() == EntityStatus.DELETED) {
            throw new ResourceNotFoundException("Conversation has been deleted");
        }

        Persona persona = personaRepository.findById(conversation.getPersonaId())
                .orElseThrow(() -> new ResourceNotFoundException("Persona not found for conversation"));

        // 1. Persist User Message
        Map<String, Object> userMeta = request.getMetadata() != null ? new HashMap<>(request.getMetadata()) : new HashMap<>();
        Message userMessage = Message.builder()
                .conversationId(conversationId)
                .senderType(SenderType.USER)
                .content(request.getContent().trim())
                .metadata(userMeta)
                .build();
        Message savedUserMessage = messageRepository.save(userMessage);

        // 2. Fetch recent conversation context for orchestrator
        List<Message> history = messageRepository.findByConversationIdOrderByCreatedAtAsc(conversationId);

        // 3. Delegate to Conversation Orchestrator abstraction
        OrchestrationContext context = OrchestrationContext.builder()
                .conversation(conversation)
                .persona(persona)
                .userMessage(savedUserMessage)
                .history(history)
                .build();

        OrchestrationResult orchestrationResult = conversationOrchestrator.orchestrateResponse(context);

        // 4. Persist Persona Message
        Message personaMessage = Message.builder()
                .conversationId(conversationId)
                .senderType(SenderType.PERSONA)
                .content(orchestrationResult.getContent())
                .metadata(orchestrationResult.getMetadata() != null ? orchestrationResult.getMetadata() : new HashMap<>())
                .build();
        Message savedPersonaMessage = messageRepository.save(personaMessage);

        // 5. Update Conversation timestamp
        conversation.setUpdatedAt(Instant.now());
        conversationRepository.save(conversation);

        log.info("Message exchange completed for conversation: [{}]", conversationId);

        return SendMessageResponse.builder()
                .userMessage(mapToResponse(savedUserMessage))
                .personaMessage(mapToResponse(savedPersonaMessage))
                .build();
    }

    @Transactional(readOnly = true)
    public List<MessageResponse> getConversationMessages(UUID conversationId) {
        if (!conversationRepository.existsById(conversationId)) {
            throw new ResourceNotFoundException("Conversation not found with id: " + conversationId);
        }

        return messageRepository.findByConversationIdOrderByCreatedAtAsc(conversationId).stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    public MessageResponse mapToResponse(Message message) {
        return MessageResponse.builder()
                .id(message.getId())
                .conversationId(message.getConversationId())
                .senderType(message.getSenderType())
                .content(message.getContent())
                .metadata(message.getMetadata())
                .createdAt(message.getCreatedAt())
                .build();
    }
}
