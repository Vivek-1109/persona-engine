package com.personaengine.app.dto.request;

import com.personaengine.app.entity.EntityStatus;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class UpdateConversationRequest {

    @Size(min = 1, max = 255, message = "Conversation title must be between 1 and 255 characters")
    private String title;

    private EntityStatus status;
}
