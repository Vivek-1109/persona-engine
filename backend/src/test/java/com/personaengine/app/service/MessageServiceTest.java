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
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class MessageServiceTest {

    @Mock
    private MessageRepository messageRepository;

    @Mock
    private ConversationRepository conversationRepository;

    @Mock
    private PersonaRepository personaRepository;

    @Mock
    private ConversationOrchestrator conversationOrchestrator;

    @InjectMocks
    private MessageService messageService;

    private UUID conversationId;
    private UUID personaId;
    private Conversation sampleConversation;
    private Persona samplePersona;

    @BeforeEach
    void setUp() {
        conversationId = UUID.randomUUID();
        personaId = UUID.randomUUID();

        samplePersona = Persona.builder()
                .id(personaId)
                .name("Alex")
                .description("Tech founder persona")
                .status(EntityStatus.ACTIVE)
                .build();

        sampleConversation = Conversation.builder()
                .id(conversationId)
                .userId(UUID.randomUUID())
                .personaId(personaId)
                .title("Initial Chat")
                .status(EntityStatus.ACTIVE)
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();
    }

    @Test
    void sendMessage_Success() {
        SendMessageRequest request = SendMessageRequest.builder()
                .content("Hello, how are you?")
                .build();

        Message savedUserMsg = Message.builder()
                .id(UUID.randomUUID())
                .conversationId(conversationId)
                .senderType(SenderType.USER)
                .content("Hello, how are you?")
                .createdAt(Instant.now())
                .build();

        Message savedPersonaMsg = Message.builder()
                .id(UUID.randomUUID())
                .conversationId(conversationId)
                .senderType(SenderType.PERSONA)
                .content("Doing great, Alex here!")
                .metadata(Map.of("modelUsed", "mock-v1"))
                .createdAt(Instant.now())
                .build();

        when(conversationRepository.findById(conversationId)).thenReturn(Optional.of(sampleConversation));
        when(personaRepository.findById(personaId)).thenReturn(Optional.of(samplePersona));
        when(messageRepository.save(any(Message.class))).thenReturn(savedUserMsg).thenReturn(savedPersonaMsg);
        when(messageRepository.findByConversationIdOrderByCreatedAtAsc(conversationId)).thenReturn(List.of(savedUserMsg));

        OrchestrationResult orchResult = OrchestrationResult.builder()
                .content("Doing great, Alex here!")
                .metadata(new HashMap<>(Map.of("modelUsed", "mock-v1")))
                .build();
        when(conversationOrchestrator.orchestrateResponse(any(OrchestrationContext.class))).thenReturn(orchResult);

        SendMessageResponse response = messageService.sendMessage(conversationId, request);

        assertThat(response).isNotNull();
        assertThat(response.getUserMessage().getContent()).isEqualTo("Hello, how are you?");
        assertThat(response.getPersonaMessage().getContent()).isEqualTo("Doing great, Alex here!");
        verify(conversationOrchestrator).orchestrateResponse(any(OrchestrationContext.class));
    }

    @Test
    void sendMessage_ConversationNotFound_ThrowsException() {
        SendMessageRequest request = SendMessageRequest.builder()
                .content("Hello")
                .build();

        when(conversationRepository.findById(conversationId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> messageService.sendMessage(conversationId, request))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void getConversationMessages_ReturnsList() {
        Message msg = Message.builder()
                .id(UUID.randomUUID())
                .conversationId(conversationId)
                .senderType(SenderType.USER)
                .content("Hello")
                .createdAt(Instant.now())
                .build();

        when(conversationRepository.existsById(conversationId)).thenReturn(true);
        when(messageRepository.findByConversationIdOrderByCreatedAtAsc(conversationId)).thenReturn(List.of(msg));

        List<MessageResponse> messages = messageService.getConversationMessages(conversationId);

        assertThat(messages).hasSize(1);
        assertThat(messages.get(0).getContent()).isEqualTo("Hello");
    }
}
