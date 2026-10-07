package com.personaengine.app.service;

import com.personaengine.app.dto.request.CreateUserRequest;
import com.personaengine.app.dto.request.UpdateUserRequest;
import com.personaengine.app.dto.response.UserResponse;
import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.entity.User;
import com.personaengine.app.exception.ConflictException;
import com.personaengine.app.exception.ResourceNotFoundException;
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
public class UserService {

    private static final Logger log = LoggerFactory.getLogger(UserService.class);

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    public UserResponse createUser(CreateUserRequest request) {
        log.info("Creating user with email: {}", request.getEmail());
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new ConflictException("User with email " + request.getEmail() + " already exists");
        }

        User user = User.builder()
                .email(request.getEmail().trim().toLowerCase())
                .name(request.getName().trim())
                .status(EntityStatus.ACTIVE)
                .build();

        User saved = userRepository.save(user);
        log.info("User created successfully with id: {}", saved.getId());
        return mapToResponse(saved);
    }

    @Transactional(readOnly = true)
    public UserResponse getUserById(UUID id) {
        User user = userRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + id));
        return mapToResponse(user);
    }

    @Transactional(readOnly = true)
    public List<UserResponse> listUsers() {
        return userRepository.findByStatusNot(EntityStatus.DELETED).stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    public UserResponse updateUser(UUID id, UpdateUserRequest request) {
        User user = userRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + id));

        if (request.getName() != null && !request.getName().isBlank()) {
            user.setName(request.getName().trim());
        }
        if (request.getStatus() != null) {
            user.setStatus(request.getStatus());
        }

        User updated = userRepository.save(user);
        log.info("User updated successfully with id: {}", updated.getId());
        return mapToResponse(updated);
    }

    public void deleteUser(UUID id) {
        User user = userRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + id));
        user.setStatus(EntityStatus.DELETED);
        userRepository.save(user);
        log.info("User deactivated with id: {}", id);
    }

    @Transactional(readOnly = true)
    public User getOrCreateDefaultUser() {
        return userRepository.findByEmail("default@personaengine.ai")
                .orElseGet(() -> {
                    User newUser = User.builder()
                            .id(UUID.fromString("00000000-0000-0000-0000-000000000001"))
                            .email("default@personaengine.ai")
                            .name("Default User")
                            .status(EntityStatus.ACTIVE)
                            .build();
                    return userRepository.save(newUser);
                });
    }

    public UserResponse mapToResponse(User user) {
        return UserResponse.builder()
                .id(user.getId())
                .email(user.getEmail())
                .name(user.getName())
                .status(user.getStatus())
                .createdAt(user.getCreatedAt())
                .updatedAt(user.getUpdatedAt())
                .build();
    }
}
