package com.personaengine.app.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.personaengine.app.dto.request.SendMessageRequest;
import com.personaengine.app.dto.response.MessageResponse;
import com.personaengine.app.dto.response.SendMessageResponse;
import com.personaengine.app.entity.SenderType;
import com.personaengine.app.service.MessageService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(MessageController.class)
class MessageControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockitoBean
    private MessageService messageService;

    @Test
    void sendMessage_Success_Returns201() throws Exception {
        UUID conversationId = UUID.randomUUID();
        SendMessageRequest request = SendMessageRequest.builder()
                .content("Hello Persona")
                .build();

        MessageResponse userMsg = MessageResponse.builder()
                .id(UUID.randomUUID())
                .conversationId(conversationId)
                .senderType(SenderType.USER)
                .content("Hello Persona")
                .createdAt(Instant.now())
                .build();

        MessageResponse personaMsg = MessageResponse.builder()
                .id(UUID.randomUUID())
                .conversationId(conversationId)
                .senderType(SenderType.PERSONA)
                .content("Hello there!")
                .createdAt(Instant.now())
                .build();

        SendMessageResponse response = SendMessageResponse.builder()
                .userMessage(userMsg)
                .personaMessage(personaMsg)
                .build();

        when(messageService.sendMessage(eq(conversationId), any(SendMessageRequest.class))).thenReturn(response);

        mockMvc.perform(post("/api/conversations/" + conversationId + "/messages")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.userMessage.content").value("Hello Persona"))
                .andExpect(jsonPath("$.personaMessage.content").value("Hello there!"));
    }

    @Test
    void sendMessage_EmptyContent_Returns400() throws Exception {
        UUID conversationId = UUID.randomUUID();
        SendMessageRequest request = SendMessageRequest.builder()
                .content("")
                .build();

        mockMvc.perform(post("/api/conversations/" + conversationId + "/messages")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("VALIDATION_ERROR"));
    }

    @Test
    void getMessages_ReturnsList() throws Exception {
        UUID conversationId = UUID.randomUUID();
        when(messageService.getConversationMessages(conversationId)).thenReturn(List.of());

        mockMvc.perform(get("/api/conversations/" + conversationId + "/messages"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray());
    }
}
