package com.personaengine.app.dto.response;

import com.personaengine.app.entity.SenderType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.Instant;
import java.util.Map;
import java.util.UUID;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MessageResponse {
    private UUID id;
    private UUID conversationId;
    private SenderType senderType;
    private String content;
    private Map<String, Object> metadata;
    private Instant createdAt;
}
