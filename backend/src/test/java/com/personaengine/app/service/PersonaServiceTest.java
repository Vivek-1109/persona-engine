package com.personaengine.app.service;

import com.personaengine.app.dto.request.CreatePersonaRequest;
import com.personaengine.app.dto.response.PersonaResponse;
import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.entity.Persona;
import com.personaengine.app.entity.User;
import com.personaengine.app.exception.ResourceNotFoundException;
import com.personaengine.app.repository.ConversationRepository;
import com.personaengine.app.repository.PersonaRepository;
import com.personaengine.app.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class PersonaServiceTest {

    @Mock
    private PersonaRepository personaRepository;

    @Mock
    private ConversationRepository conversationRepository;

    @Mock
    private UserRepository userRepository;

    @Mock
    private UserService userService;

    @InjectMocks
    private PersonaService personaService;

    private UUID personaId;
    private UUID userId;
    private Persona samplePersona;

    @BeforeEach
    void setUp() {
        personaId = UUID.randomUUID();
        userId = UUID.randomUUID();
        samplePersona = Persona.builder()
                .id(personaId)
                .userId(userId)
                .name("Alex")
                .description("Analytical tech mind")
                .status(EntityStatus.ACTIVE)
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();
    }

    @Test
    void createPersona_WithValidUser_Success() {
        CreatePersonaRequest request = CreatePersonaRequest.builder()
                .userId(userId)
                .name("Alex")
                .description("Analytical tech mind")
                .build();

        when(userRepository.existsById(userId)).thenReturn(true);
        when(personaRepository.save(any(Persona.class))).thenReturn(samplePersona);
        when(conversationRepository.countByPersonaIdAndStatusNot(personaId, EntityStatus.DELETED)).thenReturn(0L);

        PersonaResponse response = personaService.createPersona(request);

        assertThat(response).isNotNull();
        assertThat(response.getName()).isEqualTo("Alex");
        verify(personaRepository).save(any(Persona.class));
    }

    @Test
    void getPersonaById_Found() {
        when(personaRepository.findById(personaId)).thenReturn(Optional.of(samplePersona));
        when(conversationRepository.countByPersonaIdAndStatusNot(personaId, EntityStatus.DELETED)).thenReturn(2L);

        PersonaResponse response = personaService.getPersonaById(personaId);

        assertThat(response.getId()).isEqualTo(personaId);
        assertThat(response.getConversationCount()).isEqualTo(2L);
    }

    @Test
    void getPersonaById_NotFound_ThrowsException() {
        when(personaRepository.findById(personaId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> personaService.getPersonaById(personaId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void listPersonas_ReturnsAll() {
        when(personaRepository.findByStatusNot(EntityStatus.DELETED)).thenReturn(List.of(samplePersona));
        when(conversationRepository.countByPersonaIdAndStatusNot(personaId, EntityStatus.DELETED)).thenReturn(1L);

        List<PersonaResponse> list = personaService.listPersonas(null);

        assertThat(list).hasSize(1);
        assertThat(list.get(0).getName()).isEqualTo("Alex");
    }
}
