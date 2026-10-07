package com.personaengine.app.service;

import com.personaengine.app.dto.request.CreatePersonaRequest;
import com.personaengine.app.dto.request.UpdatePersonaRequest;
import com.personaengine.app.dto.response.PersonaResponse;
import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.entity.Persona;
import com.personaengine.app.entity.User;
import com.personaengine.app.exception.ResourceNotFoundException;
import com.personaengine.app.repository.ConversationRepository;
import com.personaengine.app.repository.PersonaRepository;
import com.personaengine.app.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
@Transactional
public class PersonaService {

    private static final Logger log = LoggerFactory.getLogger(PersonaService.class);

    private final PersonaRepository personaRepository;
    private final ConversationRepository conversationRepository;
    private final UserRepository userRepository;
    private final UserService userService;

    public PersonaService(PersonaRepository personaRepository,
                          ConversationRepository conversationRepository,
                          UserRepository userRepository,
                          UserService userService) {
        this.personaRepository = personaRepository;
        this.conversationRepository = conversationRepository;
        this.userRepository = userRepository;
        this.userService = userService;
    }

    public PersonaResponse createPersona(CreatePersonaRequest request) {
        log.info("Creating persona: [{}]", request.getName());

        UUID targetUserId = request.getUserId();
        if (targetUserId == null) {
            User defaultUser = userService.getOrCreateDefaultUser();
            targetUserId = defaultUser.getId();
        } else {
            if (!userRepository.existsById(targetUserId)) {
                throw new ResourceNotFoundException("User not found with id: " + targetUserId);
            }
        }

        Persona persona = Persona.builder()
                .userId(targetUserId)
                .name(request.getName().trim())
                .description(request.getDescription() != null ? request.getDescription().trim() : null)
                .status(EntityStatus.ACTIVE)
                .build();

        Persona saved = personaRepository.save(persona);
        log.info("Persona created successfully with id: {}", saved.getId());
        return mapToResponse(saved);
    }

    @Transactional(readOnly = true)
    public PersonaResponse getPersonaById(UUID id) {
        Persona persona = personaRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Persona not found with id: " + id));
        return mapToResponse(persona);
    }

    @Transactional(readOnly = true)
    public List<PersonaResponse> listPersonas(UUID userId) {
        List<Persona> personas;
        if (userId != null) {
            personas = personaRepository.findByUserIdAndStatusNot(userId, EntityStatus.DELETED);
        } else {
            personas = personaRepository.findByStatusNot(EntityStatus.DELETED);
        }

        return personas.stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    public PersonaResponse updatePersona(UUID id, UpdatePersonaRequest request) {
        Persona persona = personaRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Persona not found with id: " + id));

        if (request.getName() != null && !request.getName().isBlank()) {
            persona.setName(request.getName().trim());
        }
        if (request.getDescription() != null) {
            persona.setDescription(request.getDescription().trim());
        }
        if (request.getStatus() != null) {
            persona.setStatus(request.getStatus());
        }

        Persona updated = personaRepository.save(persona);
        log.info("Persona updated successfully with id: {}", updated.getId());
        return mapToResponse(updated);
    }

    public void deletePersona(UUID id) {
        Persona persona = personaRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Persona not found with id: " + id));
        persona.setStatus(EntityStatus.DELETED);
        personaRepository.save(persona);
        log.info("Persona soft-deleted with id: {}", id);
    }

    public PersonaResponse mapToResponse(Persona persona) {
        long conversationCount = conversationRepository.countByPersonaIdAndStatusNot(persona.getId(), EntityStatus.DELETED);
        return PersonaResponse.builder()
                .id(persona.getId())
                .userId(persona.getUserId())
                .name(persona.getName())
                .description(persona.getDescription())
                .status(persona.getStatus())
                .conversationCount(conversationCount)
                .createdAt(persona.getCreatedAt())
                .updatedAt(persona.getUpdatedAt())
                .build();
    }
}
