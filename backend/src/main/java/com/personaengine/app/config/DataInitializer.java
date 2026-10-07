package com.personaengine.app.config;

import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.entity.Persona;
import com.personaengine.app.entity.User;
import com.personaengine.app.repository.PersonaRepository;
import com.personaengine.app.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.util.UUID;

@Component
@Profile("!test")
public class DataInitializer implements CommandLineRunner {

    private static final Logger log = LoggerFactory.getLogger(DataInitializer.class);

    private final UserRepository userRepository;
    private final PersonaRepository personaRepository;

    public DataInitializer(UserRepository userRepository, PersonaRepository personaRepository) {
        this.userRepository = userRepository;
        this.personaRepository = personaRepository;
    }

    @Override
    public void run(String... args) {
        if (userRepository.count() == 0) {
            log.info("Initializing baseline demo user and personas for Persona Engine...");

            User demoUser = User.builder()
                    .id(UUID.fromString("00000000-0000-0000-0000-000000000001"))
                    .email("demo@personaengine.ai")
                    .name("Demo User")
                    .status(EntityStatus.ACTIVE)
                    .build();
            userRepository.save(demoUser);

            Persona techPersona = Persona.builder()
                    .userId(demoUser.getId())
                    .name("Alex (Startup Founder)")
                    .description("Fast-paced, analytical, tech-savvy, direct communication style with startup analogies.")
                    .status(EntityStatus.ACTIVE)
                    .build();
            personaRepository.save(techPersona);

            Persona creativePersona = Persona.builder()
                    .userId(demoUser.getId())
                    .name("Maya (Creative Storyteller)")
                    .description("Empathetic, expressive, vivid conversationalist with thoughtful humor and deep questions.")
                    .status(EntityStatus.ACTIVE)
                    .build();
            personaRepository.save(creativePersona);

            log.info("Baseline seed completed: 1 demo user and 2 baseline personas created.");
        }
    }
}
