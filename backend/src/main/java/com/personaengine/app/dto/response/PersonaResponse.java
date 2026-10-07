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
public class PersonaResponse {
    private UUID id;
    private UUID userId;
    private String name;
    private String description;
    private EntityStatus status;
    private long conversationCount;
    private Instant createdAt;
    private Instant updatedAt;
}
