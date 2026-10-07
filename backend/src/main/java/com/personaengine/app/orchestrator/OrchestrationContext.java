package com.personaengine.app.orchestrator;

import com.personaengine.app.entity.Conversation;
import com.personaengine.app.entity.Message;
import com.personaengine.app.entity.Persona;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class OrchestrationContext {
    private Conversation conversation;
    private Persona persona;
    private Message userMessage;
    private List<Message> history;
}
