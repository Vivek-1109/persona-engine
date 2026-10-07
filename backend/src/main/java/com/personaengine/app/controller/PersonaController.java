package com.personaengine.app.controller;

import com.personaengine.app.dto.request.CreatePersonaRequest;
import com.personaengine.app.dto.request.UpdatePersonaRequest;
import com.personaengine.app.dto.response.PersonaResponse;
import com.personaengine.app.service.PersonaService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.UUID;

@RestController
@RequestMapping("/api/personas")
public class PersonaController {

    private final PersonaService personaService;

    public PersonaController(PersonaService personaService) {
        this.personaService = personaService;
    }

    @PostMapping
    public ResponseEntity<PersonaResponse> createPersona(@Valid @RequestBody CreatePersonaRequest request) {
        PersonaResponse response = personaService.createPersona(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping
    public ResponseEntity<List<PersonaResponse>> listPersonas(
            @RequestParam(name = "userId", required = false) UUID userId) {
        return ResponseEntity.ok(personaService.listPersonas(userId));
    }

    @GetMapping("/{id}")
    public ResponseEntity<PersonaResponse> getPersona(@PathVariable UUID id) {
        return ResponseEntity.ok(personaService.getPersonaById(id));
    }

    @PatchMapping("/{id}")
    public ResponseEntity<PersonaResponse> updatePersona(
            @PathVariable UUID id,
            @Valid @RequestBody UpdatePersonaRequest request) {
        return ResponseEntity.ok(personaService.updatePersona(id, request));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deletePersona(@PathVariable UUID id) {
        personaService.deletePersona(id);
        return ResponseEntity.noContent().build();
    }
}
