# Persona Engine

> **Personalized Conversational AI Foundation (Phase 1)**

Persona Engine is an AI architecture aimed at learning and reproducing a specific person's conversational behavior, communication style, sentence structures, humor, sarcasm, emotional reactions, and long-term memory from conversation data.

**Phase 1** establishes the clean application foundation: a modern React frontend, a decoupled Spring Boot 3.4 REST backend, Flyway-migrated PostgreSQL schema, and mock abstraction layers for future AI intelligence (Conversation Orchestrator, Model Gateway, Personality Engine, and Memory Engine).

---

## Architecture At a Glance

```text
Frontend (React 19 + TypeScript + Tailwind CSS)
    │
    ▼ REST APIs (JSON DTOs)
Backend (Spring Boot 3.4 + Java 21/22)
    │
    ├──▶ PostgreSQL (Versioned via Flyway V1-V4)
    │
    └──▶ Conversation Orchestrator Abstraction
             ├──▶ Personality Engine (Stub)
             ├──▶ Memory Engine (Stub)
             ├──▶ Topic Engine (Stub)
             └──▶ Model Gateway (MockModelGateway)
```

---

## Repository Structure

```text
persona-engine/
│
├── frontend/                # React 19 + TypeScript + Vite + Tailwind CSS v4
│   ├── src/
│   │   ├── components/      # UI components (Navbar, Modal, StatusBadge)
│   │   ├── features/        # Feature domains (dashboard, personas, conversations, chat)
│   │   ├── services/        # Dedicated API service layer (apiClient, personaApi, ...)
│   │   └── types/           # Domain TypeScript interfaces
│
├── backend/                 # Spring Boot 3.4 Maven application
│   └── src/
│       ├── main/java/com/personaengine/app/
│       │   ├── config/      # CORS and Baseline Seed Data Initializer
│       │   ├── controller/  # REST Controllers (User, Persona, Conversation, Message)
│       │   ├── dto/         # Request & Response Data Transfer Objects
│       │   ├── entity/      # JPA Entities (User, Persona, Conversation, Message)
│       │   ├── exception/   # Centralized GlobalExceptionHandler & custom exceptions
│       │   ├── future/      # Contracts for future ML components
│       │   ├── gateway/     # ModelGateway & MockModelGateway
│       │   ├── orchestrator/# ConversationOrchestrator & MockConversationOrchestrator
│       │   ├── repository/  # Spring Data JPA Repositories
│       │   └── service/     # Business logic layer
│       └── main/resources/
│           ├── application.yml
│           └── db/migration/# Flyway migrations (V1, V2, V3, V4)
│
├── docs/                    # Architecture, API, and Database specifications
│   ├── architecture.md
│   ├── api.md
│   └── database.md
│
├── docker-compose.yml       # PostgreSQL 16 container definition
├── .env.example             # Configuration reference
├── PROJECT_VISION.md        # Architectural north star specification
└── README.md
```

---

## Quick Start & Local Setup

### Prerequisites
- **Java 21** or **22**
- **Maven 3.9+**
- **Node.js 20+** & **npm 10+**
- **PostgreSQL 16** (or Docker Desktop)

---

### Step 1: Start PostgreSQL

#### Option A: Using Docker Compose
```bash
docker compose up -d
```

#### Option B: Using Local PostgreSQL
Ensure PostgreSQL is running locally on port 5432 and create the database:
```sql
CREATE DATABASE persona_engine;
```

---

### Step 2: Configure Environment

Copy `.env.example` to your environment or configure your credentials:
```bash
cp .env.example .env
```
Default database connection:
- URL: `jdbc:postgresql://localhost:5432/persona_engine`
- Username: `postgres`
- Password: `postgres`

---

### Step 3: Run the Backend

```bash
cd backend
mvn spring-boot:run
```
The backend starts on `http://localhost:8080`.
Flyway will automatically execute migrations `V1` through `V4`, and `DataInitializer` will seed a default user and sample personas.

To run the automated backend test suite:
```bash
cd backend
mvn test
```

---

### Step 4: Run the Frontend

```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## Definition of Done Verification (Phase 1)

- [x] **Frontend**:
  - Dashboard with system statistics and active personas
  - Persona creation, editing, deletion, and conversation counter
  - Conversation creation, listing, filtering, and deletion
  - Full-featured chat interface with optimistic updates, loading states, and error handling
  - Dedicated API service layer (`services/`) with no direct DB/model coupling
- [x] **Backend**:
  - Spring Boot 3.4.3 layered architecture (Controllers, Services, Repositories, DTOs)
  - Flyway migrations for PostgreSQL (`users`, `personas`, `conversations`, `messages`)
  - Strict input validation (`@Valid`, `@NotBlank`, size constraints)
  - Centralized exception handling (`@RestControllerAdvice`) returning uniform `ErrorResponse`
  - `ConversationOrchestrator` abstraction with `MockConversationOrchestrator`
  - `ModelGateway` abstraction with `MockModelGateway`
  - 20 automated tests passing with zero failures
- [x] **Database**:
  - PostgreSQL schema with foreign keys and cascading deletes
  - Query indexes on `user_id`, `persona_id`, `created_at`, `status`
  - `JSONB` metadata on `messages` for flexible future ML parameter storage
- [x] **Architecture**:
  - Future AI engine contracts defined (`PersonalityEngine`, `MemoryEngine`, `TopicEngine`, `ConversationAnalyzer`)
  - Replaceable model gateway (no commercial API lock-in)
  - Full documentation (`docs/architecture.md`, `docs/api.md`, `docs/database.md`, `PROJECT_VISION.md`)
