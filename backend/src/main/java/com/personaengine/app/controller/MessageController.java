package com.personaengine.app.controller;

import com.personaengine.app.dto.request.SendMessageRequest;
import com.personaengine.app.dto.response.MessageResponse;
import com.personaengine.app.dto.response.SendMessageResponse;
import com.personaengine.app.service.MessageService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/conversations/{conversationId}/messages")
public class MessageController {

    private final MessageService messageService;

    public MessageController(MessageService messageService) {
        this.messageService = messageService;
    }

    @PostMapping
    public ResponseEntity<SendMessageResponse> sendMessage(
            @PathVariable UUID conversationId,
            @Valid @RequestBody SendMessageRequest request) {
        SendMessageResponse response = messageService.sendMessage(conversationId, request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping
    public ResponseEntity<List<MessageResponse>> getMessages(@PathVariable UUID conversationId) {
        return ResponseEntity.ok(messageService.getConversationMessages(conversationId));
    }
}
