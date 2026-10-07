package com.personaengine.app.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.personaengine.app.dto.request.CreatePersonaRequest;
import com.personaengine.app.dto.response.PersonaResponse;
import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.exception.ResourceNotFoundException;
import com.personaengine.app.service.PersonaService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(PersonaController.class)
class PersonaControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockitoBean
    private PersonaService personaService;

    @Test
    void createPersona_Success_Returns201() throws Exception {
        UUID id = UUID.randomUUID();
        CreatePersonaRequest request = CreatePersonaRequest.builder()
                .name("Alex")
                .description("Founder style")
                .build();

        PersonaResponse response = PersonaResponse.builder()
                .id(id)
                .name("Alex")
                .description("Founder style")
                .status(EntityStatus.ACTIVE)
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();

        when(personaService.createPersona(any(CreatePersonaRequest.class))).thenReturn(response);

        mockMvc.perform(post("/api/personas")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.id").value(id.toString()))
                .andExpect(jsonPath("$.name").value("Alex"));
    }

    @Test
    void createPersona_BlankName_Returns400ValidationError() throws Exception {
        CreatePersonaRequest request = CreatePersonaRequest.builder()
                .name("")
                .build();

        mockMvc.perform(post("/api/personas")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("VALIDATION_ERROR"));
    }

    @Test
    void getPersona_NotFound_Returns404() throws Exception {
        UUID id = UUID.randomUUID();
        when(personaService.getPersonaById(id)).thenThrow(new ResourceNotFoundException("Persona not found with id: " + id));

        mockMvc.perform(get("/api/personas/" + id))
                .andExpect(status().isNotFound())
                .andExpect(jsonPath("$.code").value("RESOURCE_NOT_FOUND"));
    }

    @Test
    void listPersonas_Returns200() throws Exception {
        when(personaService.listPersonas(null)).thenReturn(List.of());

        mockMvc.perform(get("/api/personas"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$").isArray());
    }
}
