package com.personaengine.app.repository;

import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.entity.Persona;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface PersonaRepository extends JpaRepository<Persona, UUID> {

    List<Persona> findByUserIdAndStatusNot(UUID userId, EntityStatus status);

    List<Persona> findByStatusNot(EntityStatus status);

    long countByUserIdAndStatusNot(UUID userId, EntityStatus status);
}
