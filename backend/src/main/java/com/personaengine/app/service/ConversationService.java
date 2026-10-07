package com.personaengine.app.service;

import com.personaengine.app.dto.request.CreateConversationRequest;
import com.personaengine.app.dto.request.UpdateConversationRequest;
import com.personaengine.app.dto.response.ConversationResponse;
import com.personaengine.app.entity.Conversation;
import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.entity.Persona;
import com.personaengine.app.entity.User;
import com.personaengine.app.exception.ResourceNotFoundException;
import com.personaengine.app.repository.ConversationRepository;
import com.personaengine.app.repository.MessageRepository;
import com.personaengine.app.repository.PersonaRepository;
import com.personaengine.app.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@Transactional
public class ConversationService {

    private static final Logger log = LoggerFactory.getLogger(ConversationService.class);

    private final ConversationRepository conversationRepository;
    private final PersonaRepository personaRepository;
    private final UserRepository userRepository;
    private final MessageRepository messageRepository;
    private final UserService userService;

    public ConversationService(ConversationRepository conversationRepository,
                               PersonaRepository personaRepository,
                               UserRepository userRepository,
                               MessageRepository messageRepository,
                               UserService userService) {
        this.conversationRepository = conversationRepository;
        this.personaRepository = personaRepository;
        this.userRepository = userRepository;
        this.messageRepository = messageRepository;
        this.userService = userService;
    }

    public ConversationResponse createConversation(CreateConversationRequest request) {
        log.info("Creating conversation with title: [{}] for persona: [{}]", request.getTitle(), request.getPersonaId());

        Persona persona = personaRepository.findById(request.getPersonaId())
                .orElseThrow(() -> new ResourceNotFoundException("Persona not found with id: " + request.getPersonaId()));

        UUID targetUserId = request.getUserId();
        if (targetUserId == null) {
            targetUserId = persona.getUserId();
        }
        if (targetUserId == null || !userRepository.existsById(targetUserId)) {
            User defaultUser = userService.getOrCreateDefaultUser();
            targetUserId = defaultUser.getId();
        }

        Conversation conversation = Conversation.builder()
                .userId(targetUserId)
                .personaId(persona.getId())
                .title(request.getTitle().trim())
                .status(EntityStatus.ACTIVE)
                .build();

        Conversation saved = conversationRepository.save(conversation);
        log.info("Conversation created successfully with id: {}", saved.getId());
        return mapToResponse(saved, persona.getName());
    }

    @Transactional(readOnly = true)
    public ConversationResponse getConversationById(UUID id) {
        Conversation conversation = conversationRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Conversation not found with id: " + id));
        String personaName = personaRepository.findById(conversation.getPersonaId())
                .map(Persona::getName)
                .orElse("Unknown Persona");
        return mapToResponse(conversation, personaName);
    }

    @Transactional(readOnly = true)
    public List<ConversationResponse> listConversations(UUID userId, UUID personaId) {
        List<Conversation> conversations;
        if (personaId != null) {
            conversations = conversationRepository.findByPersonaIdAndStatusNotOrderByCreatedAtDesc(personaId, EntityStatus.DELETED);
        } else if (userId != null) {
            conversations = conversationRepository.findByUserIdAndStatusNotOrderByCreatedAtDesc(userId, EntityStatus.DELETED);
        } else {
            conversations = conversationRepository.findByStatusNotOrderByCreatedAtDesc(EntityStatus.DELETED);
        }

        return conversations.stream()
                .map(conv -> {
                    String personaName = personaRepository.findById(conv.getPersonaId())
                            .map(Persona::getName)
                            .orElse("Unknown Persona");
                    return mapToResponse(conv, personaName);
                })
                .collect(Collectors.toList());
    }

    public ConversationResponse updateConversation(UUID id, UpdateConversationRequest request) {
        Conversation conversation = conversationRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Conversation not found with id: " + id));

        if (request.getTitle() != null && !request.getTitle().isBlank()) {
            conversation.setTitle(request.getTitle().trim());
        }
        if (request.getStatus() != null) {
            conversation.setStatus(request.getStatus());
        }

        Conversation updated = conversationRepository.save(conversation);
        String personaName = personaRepository.findById(updated.getPersonaId())
                .map(Persona::getName)
                .orElse("Unknown Persona");
        log.info("Conversation updated successfully with id: {}", updated.getId());
        return mapToResponse(updated, personaName);
    }

    public void deleteConversation(UUID id) {
        Conversation conversation = conversationRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Conversation not found with id: " + id));
        conversation.setStatus(EntityStatus.DELETED);
        conversationRepository.save(conversation);
        log.info("Conversation soft-deleted with id: {}", id);
    }

    public ConversationResponse mapToResponse(Conversation conversation, String personaName) {
        long messageCount = messageRepository.countByConversationId(conversation.getId());
        return ConversationResponse.builder()
                .id(conversation.getId())
                .userId(conversation.getUserId())
                .personaId(conversation.getPersonaId())
                .personaName(personaName)
                .title(conversation.getTitle())
                .status(conversation.getStatus())
                .messageCount(messageCount)
                .createdAt(conversation.getCreatedAt())
                .updatedAt(conversation.getUpdatedAt())
                .build();
    }
}
