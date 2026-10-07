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
public class UserResponse {
    private UUID id;
    private String email;
    private String name;
    private EntityStatus status;
    private Instant createdAt;
    private Instant updatedAt;
}
