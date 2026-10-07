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
public class UpdatePersonaRequest {

    @Size(min = 2, max = 100, message = "Persona name must be between 2 and 100 characters")
    private String name;

    @Size(max = 2000, message = "Description cannot exceed 2000 characters")
    private String description;

    private EntityStatus status;
}
