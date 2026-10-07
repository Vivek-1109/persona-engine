package com.personaengine.app.orchestrator;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.HashMap;
import java.util.Map;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OrchestrationResult {
    private String content;
    @Builder.Default
    private Map<String, Object> metadata = new HashMap<>();
}
