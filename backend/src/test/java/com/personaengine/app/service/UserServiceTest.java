package com.personaengine.app.service;

import com.personaengine.app.dto.request.CreateUserRequest;
import com.personaengine.app.dto.request.UpdateUserRequest;
import com.personaengine.app.dto.response.UserResponse;
import com.personaengine.app.entity.EntityStatus;
import com.personaengine.app.entity.User;
import com.personaengine.app.exception.ConflictException;
import com.personaengine.app.exception.ResourceNotFoundException;
import com.personaengine.app.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class UserServiceTest {

    @Mock
    private UserRepository userRepository;

    @InjectMocks
    private UserService userService;

    private User sampleUser;
    private UUID userId;

    @BeforeEach
    void setUp() {
        userId = UUID.randomUUID();
        sampleUser = User.builder()
                .id(userId)
                .email("test@example.com")
                .name("Test User")
                .status(EntityStatus.ACTIVE)
                .createdAt(Instant.now())
                .updatedAt(Instant.now())
                .build();
    }

    @Test
    void createUser_Success() {
        CreateUserRequest request = CreateUserRequest.builder()
                .email("new@example.com")
                .name("New User")
                .build();

        when(userRepository.existsByEmail("new@example.com")).thenReturn(false);
        when(userRepository.save(any(User.class))).thenReturn(sampleUser);

        UserResponse response = userService.createUser(request);

        assertThat(response).isNotNull();
        assertThat(response.getEmail()).isEqualTo("test@example.com");
        verify(userRepository).save(any(User.class));
    }

    @Test
    void createUser_DuplicateEmail_ThrowsConflictException() {
        CreateUserRequest request = CreateUserRequest.builder()
                .email("test@example.com")
                .name("Another Name")
                .build();

        when(userRepository.existsByEmail("test@example.com")).thenReturn(true);

        assertThatThrownBy(() -> userService.createUser(request))
                .isInstanceOf(ConflictException.class)
                .hasMessageContaining("already exists");
    }

    @Test
    void getUserById_Found() {
        when(userRepository.findById(userId)).thenReturn(Optional.of(sampleUser));

        UserResponse response = userService.getUserById(userId);

        assertThat(response.getId()).isEqualTo(userId);
        assertThat(response.getName()).isEqualTo("Test User");
    }

    @Test
    void getUserById_NotFound_ThrowsException() {
        when(userRepository.findById(userId)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> userService.getUserById(userId))
                .isInstanceOf(ResourceNotFoundException.class);
    }

    @Test
    void updateUser_Success() {
        when(userRepository.findById(userId)).thenReturn(Optional.of(sampleUser));
        when(userRepository.save(any(User.class))).thenReturn(sampleUser);

        UpdateUserRequest request = UpdateUserRequest.builder()
                .name("Updated Name")
                .build();

        UserResponse response = userService.updateUser(userId, request);

        assertThat(response).isNotNull();
        verify(userRepository).save(sampleUser);
    }
}
