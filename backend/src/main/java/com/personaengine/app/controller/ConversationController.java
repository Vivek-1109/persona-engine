package com.personaengine.app.controller;

import com.personaengine.app.dto.request.CreateConversationRequest;
import com.personaengine.app.dto.request.UpdateConversationRequest;
import com.personaengine.app.dto.response.ConversationResponse;
import com.personaengine.app.service.ConversationService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/conversations")
public class ConversationController {

    private final ConversationService conversationService;

    public ConversationController(ConversationService conversationService) {
        this.conversationService = conversationService;
    }

    @PostMapping
    public ResponseEntity<ConversationResponse> createConversation(@Valid @RequestBody CreateConversationRequest request) {
        ConversationResponse response = conversationService.createConversation(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping
    public ResponseEntity<List<ConversationResponse>> listConversations(
            @RequestParam(name = "userId", required = false) UUID userId,
            @RequestParam(name = "personaId", required = false) UUID personaId) {
        return ResponseEntity.ok(conversationService.listConversations(userId, personaId));
    }

    @GetMapping("/{id}")
    public ResponseEntity<ConversationResponse> getConversation(@PathVariable UUID id) {
        return ResponseEntity.ok(conversationService.getConversationById(id));
    }

    @PatchMapping("/{id}")
    public ResponseEntity<ConversationResponse> updateConversation(
            @PathVariable UUID id,
            @Valid @RequestBody UpdateConversationRequest request) {
        return ResponseEntity.ok(conversationService.updateConversation(id, request));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteConversation(@PathVariable UUID id) {
        conversationService.deleteConversation(id);
        return ResponseEntity.noContent().build();
    }
}
