package com.personaengine.app.gateway;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ModelRequest {
    private UUID conversationId;
    private UUID personaId;
    private String personaName;
    private String personaDescription;
    private String userMessage;
    private List<String> recentHistory;
    @Builder.Default
    private Map<String, Object> context = new HashMap<>();
}
