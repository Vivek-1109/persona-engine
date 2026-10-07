package com.personaengine.app.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.UUID;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CreateConversationRequest {

    private UUID userId;

    @NotNull(message = "Persona ID is required")
    private UUID personaId;

    @NotBlank(message = "Conversation title is required")
    @Size(min = 1, max = 255, message = "Conversation title must be between 1 and 255 characters")
    private String title;
}
