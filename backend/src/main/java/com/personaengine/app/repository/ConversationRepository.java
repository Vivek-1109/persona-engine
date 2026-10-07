package com.personaengine.app.repository;

import com.personaengine.app.entity.Conversation;
import com.personaengine.app.entity.EntityStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.UUID;

@Repository
public interface ConversationRepository extends JpaRepository<Conversation, UUID> {

    List<Conversation> findByUserIdAndStatusNotOrderByCreatedAtDesc(UUID userId, EntityStatus status);

    List<Conversation> findByPersonaIdAndStatusNotOrderByCreatedAtDesc(UUID personaId, EntityStatus status);

    List<Conversation> findByStatusNotOrderByCreatedAtDesc(EntityStatus status);

    long countByPersonaIdAndStatusNot(UUID personaId, EntityStatus status);
}
