package com.personaengine.app.dto.request;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Map;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SendMessageRequest {

    @NotBlank(message = "Message content cannot be empty")
    @Size(min = 1, max = 10000, message = "Message content must not exceed 10000 characters")
    private String content;

    private Map<String, Object> metadata;
}
