package com.personaengine.app.dto.response;

import com.personaengine.app.entity.EntityStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.Instant;
import java.util.UUID;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ConversationResponse {
    private UUID id;
    private UUID userId;
    private UUID personaId;
    private String personaName;
    private String title;
    private EntityStatus status;
    private long messageCount;
    private Instant createdAt;
    private Instant updatedAt;
}
