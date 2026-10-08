# Stage 6C — Memory Engine Report

## 1. Implementation

Stage 6C implements the persistent memory layer for the Persona Engine. The implementation is modular, deterministic, fully tested, and preserves all previous stage baselines without modification.

### Files Created:
1. `ml/src/memory/memory_schema.py` — Strongly typed Pydantic models (`Memory`, `MemoryType`, `ImportanceLevel`, `MemoryStatus`).
2. `ml/src/memory/memory_extractor.py` — Deterministic pattern-based extractor for facts, preferences, goals, plans, relationships, experiences, and events with noise filtering.
3. `ml/src/memory/embedding.py` — `BaseEmbeddingModel` abstract interface, 64-dimensional `DeterministicMockEmbedding`, and `cosine_similarity`.
4. `ml/src/memory/memory_store.py` — `BaseMemoryStore` interface, `InMemoryMemoryStore` repository with duplicate normalization and conflict supersession.
5. `ml/src/memory/memory_ranker.py` — Multi-factor explainable scoring engine combining semantic, topic, importance, and recency signals.
6. `ml/src/memory/memory_retriever.py` — Top-K retrieval engine consuming queries, dialogues, and Stage 6B `ContextAnalysis`.
7. `ml/src/memory/__init__.py` — Clean public package exports.
8. `ml/src/memory/README.md` — Architectural and API documentation.
9. `ml/tests/test_memory_engine.py` — Comprehensive unit test suite covering Scenarios A through Q and lifecycle CRUD.
10. `ml/scripts/test_memory_engine.py` — Multi-scenario integration benchmark executing realistic conversation tests.
11. `ml/data/stage5/reports/stage6c_memory_engine_report.md` — Formal milestone report.

### Files Modified:
None outside `ml/src/memory/` and new test/benchmark scripts. Stage 5 artifacts, Stage 6A inference service, Stage 6B context engine, model registry, and existing tests remain strictly intact.

---

## 2. Memory Taxonomy

The Memory Engine enforces a controlled taxonomy via the `MemoryType` enum:

- `fact`: Enduring ground truths about identity, education, residence, hardware (e.g., `"Studies B.Tech CSE"`, `"Lives in Delhi"`).
- `preference`: Explicit user likes, dislikes, or technology choices (e.g., `"Prefers Java"`, `"Likes gaming"`, `"Likes Marvel movies"`).
- `goal`: Active career aspirations and learning paths (e.g., `"Wants to become a backend developer"`, `"Learning Spring Boot"`).
- `plan`: Scheduled upcoming events or temporal intentions (e.g., `"Has an upcoming interview"`, `"Planning to visit Delhi next week"`).
- `relationship`: Social, academic, or professional connections (e.g., `"Rahul is a college friend"`, `"Has a sister"`).
- `experience`: Past accomplishments, hackathons, and tournaments (e.g., `"Participated in Smart India Hackathon (SIH)"`).
- `event`: Life milestones (e.g., `"Internship started"`).

Arbitrary, unvetted types are disallowed.

---

## 3. Extraction

Extraction is handled deterministically by `MemoryExtractor` using explainable rule sets:
- **Conversation History $\neq$ Persistent Memory**: Ephemeral acknowledgments (`"haan"`, `"thik hai"`, `"ok"`), momentary status checks (`"kya chal raha hai"`), and immediate invitations (`"aaja game khelte hain"`, `"kal milte hain"`, `"canteen chale?"`) yield **0 candidate memories**.
- **Explicit Surface Evidence**: Statements are matched against regular expressions targeting Hindi, Hinglish, and English phrasing.
- **Canonical Representation**: Raw utterances map to clean canonical strings (e.g., `"Mujhe Marvel movies bahut pasand hain"` maps to canonical `"Likes Marvel movies"`).
- **Conservative Extraction**: Speculative inferences are prohibited. Conversational statements like `"Java use kar raha hu"` are never misconstrued as `"Prefers Java"`.

---

## 4. Importance

Importance scoring assigns deterministic values using the `ImportanceLevel` enum:

| Level | Value | Criteria | Examples |
| :--- | :--- | :--- | :--- |
| `LOW` | `0.2` | Temporary plans, near-term schedules, one-off events | `"Has an upcoming interview"` |
| `MEDIUM` | `0.6` | Stable facts, hackathon experiences, secondary preferences | `"Participated in SIH"`, `"College is in Noida"` |
| `HIGH` | `1.0` | Core identity facts, long-term career goals, primary preferences | `"Studies B.Tech CSE"`, `"Wants to become a backend developer"`, `"Prefers Java"` |

---

## 5. Duplicate Handling

Duplicate detection prevents storage bloat through `find_duplicates()` in `BaseMemoryStore`:
1. **Exact Match**: Direct equality check on normalized strings.
2. **Normalized Content Match**: `normalize_content_text()` removes punctuation, stop words (`really`, `very`, `bahut`, `hai`, `hain`, etc.), subject pronouns (`i`, `me`, `my`, `user`), and normalizes verbs (`likes` $\rightarrow$ `like`). Example: `"I really like Java."` and `"Likes Java"` normalize to `"like java"`.
3. **Semantic Similarity Match**: High cosine similarity ($\ge 0.98$) for identical memory types.

**Action on Duplicate**:
The existing memory record is retained, `updated_at` is refreshed, and importance is promoted if the new candidate possesses a higher level. No duplicate row is inserted.

---

## 6. Conflict Handling

Conflicting memories represent competing states of the same slot (e.g., `"Prefers Java"` vs `"Prefers Python"`, or `"College in Noida"` vs `"College in Delhi"`):
- Exclusive slots are tracked through deterministic regex mapping (`CONFLICT_SLOTS`).
- When a candidate memory collides with an existing active memory in the same slot:
  1. The existing memory is transitioned to `MemoryStatus.SUPERSEDED`.
  2. The existing memory's `superseded_by` field is populated with the candidate's ID.
  3. The new candidate memory is stored with `MemoryStatus.ACTIVE`.
- Historical lineage is fully retained while active queries only surface the current valid state.

---

## 7. Retrieval

`MemoryRetriever` orchestrates multi-factor memory candidate retrieval:
- **Strict Persona Isolation**: Every query filters by `persona_id`. Memories belonging to user A are never accessible or returned for user B.
- **Active State Filtering**: Superseded and archived memories are excluded from production retrieval.
- **Context Awareness**: Retrieves memories based on user queries, multi-turn dialogue histories, or Stage 6B context intelligence.
- **Access Tracking**: `last_accessed_at` is updated whenever memories are pulled into active prompt context.

---

## 8. Ranking

`MemoryRanker` scores candidate memories using an explainable weighted linear combination:

$$\text{Final Score} = (0.40 \times S_{\text{semantic}}) + (0.30 \times S_{\text{topic}}) + (0.20 \times S_{\text{importance}}) + (0.10 \times S_{\text{recency}})$$

### Component Weights:
1. **Semantic Relevance ($0.40$)**: Vector cosine similarity between query embedding and memory embedding (with token overlap fallback).
2. **Topic Relevance ($0.30$)**: $1.0$ if memory topic aligns with context topic or query contains domain trigger keywords; $0.0$ otherwise.
3. **Importance Weight ($0.20$)**: High $= 1.0$, Medium $= 0.6$, Low $= 0.2$.
4. **Recency Weight ($0.10$)**: Exponential decay over age in days ($e^{-\Delta t / 30}$).

Every retrieval output includes an explainable `MemoryScoreBreakdown` describing each component's contribution.

---

## 9. Storage

The storage interface is decoupled behind `BaseMemoryStore` with standard CRUD and query primitives:
- `save(memory)`
- `get(memory_id)`
- `update(memory)`
- `delete(memory_id)`
- `list_by_persona(persona_id, status)`
- `search_by_persona(persona_id, query, top_k)`
- `update_last_accessed(memory_id)`
- `find_duplicates(memory)`
- `find_conflicts(memory)`
- `count(persona_id)`

`InMemoryMemoryStore` provides an in-memory implementation for automated testing and isolated local execution.

---

## 10. Embeddings

Vector representations are decoupled behind the `BaseEmbeddingModel` interface:
- **Deterministic Mock Embedding**: `DeterministicMockEmbedding` generates deterministic 64-dimensional float vectors derived from token hashes. It eliminates external model downloads, GPU dependencies, and network requirements during testing.
- **Cosine Similarity**: Mathematically sound normalized dot-product clamped between $0.0$ and $1.0$.
- Future embeddings (e.g. `all-MiniLM-L6-v2`) implement `BaseEmbeddingModel` without altering the storage or retrieval layers.

---

## 11. Context Integration

The Memory Engine consumes Stage 6B `ContextAnalysis` without modifying Context Engine code:
- **Topic Alignment**: Extracts `context_analysis.topic` (e.g., `gaming`, `college`, `technology`, `movies`) to activate topic score boosts ($+0.30$).
- **Query Resolution**: When no raw query string is supplied, `context_analysis.last_user_message` is automatically used as the retrieval anchor.
- **Disambiguation**: For ambiguous messages such as `"Aaja"`, topic signals from `ContextAnalysis` ensure domain-appropriate memory retrieval (e.g., gaming memories when gaming context is active, college memories when canteen/college context is active).

---

## 12. Test Results

### Memory Engine Unit Tests (`ml/tests/test_memory_engine.py`)
All 20 tests passed:
- `test_a_fact_extraction` — PASS
- `test_b_preference_extraction` — PASS
- `test_c_goal_extraction` — PASS
- `test_d_plan_extraction` — PASS
- `test_e_relationship_extraction` — PASS
- `test_f_experience_extraction` — PASS
- `test_g_non_memory_message` — PASS
- `test_h_non_memory_invitation` — PASS
- `test_i_duplicate_detection` — PASS
- `test_j_conflict_handling` — PASS
- `test_k_importance_scoring` — PASS
- `test_l_retrieval_relevance` — PASS
- `test_m_importance_ranking` — PASS
- `test_n_recency_ranking` — PASS
- `test_o_persona_isolation` — PASS
- `test_p_empty_memory_store` — PASS
- `test_q_context_integration` — PASS
- `test_store_crud_and_access` — PASS
- `test_extract_from_conversation` — PASS
- `test_memory_serialization` — PASS

### Integration Benchmark (`ml/scripts/test_memory_engine.py`)
Executed 5 realistic dialogue benchmarks:
- **Benchmark 1 (Gaming)**: `"bhai bgmi khelega?"` $\rightarrow$ Gaming memories ranked top-1 and top-2 (`Score=0.6000`, `Score=0.5200`). **PASS**
- **Benchmark 2 (College)**: `"Dean ne attendance ka notice nikala hai"` $\rightarrow$ College memories ranked top-1 and top-2 (`Score=0.6000`). **PASS**
- **Benchmark 3 (Technology)**: `"bhai battery bahut jaldi drain ho rahi"` $\rightarrow$ Technology memories ranked top-1 and top-2 (`Score=0.6000`, `Score=0.5200`). **PASS**
- **Benchmark 4 (Movies)**: `"Marvel ki new movie dekhi?"` $\rightarrow$ Movie preference ranked top-1 (`Score=0.6614`). **PASS**
- **Benchmark 5 (Disambiguation)**: `"Aaja"` under Gaming context yielded gaming memories as top-1; under College context yielded college memories as top-1. **PASS**

### Full Repository Regression
```
Ran 108 tests in 1.914s
OK (108/108 passing, 0 failures, 0 errors)
```
- 88 baseline tests (Stage 4, Stage 5, Stage 6A, Stage 6B)
- 20 Stage 6C Memory Engine tests

---

## 13. Privacy

- **Zero Raw Data Committed**: No private chat logs, personal identifiers, credentials, or API keys were added.
- **Synthetic Test Fixtures**: All unit tests and benchmarks operate strictly on synthetic persona items.
- **No Embedding Dumps**: Vector arrays are omitted from logs and console outputs.

---

## 14. Limitations

1. **Rule-Based Extraction**: Relies on curated regex patterns. Complex indirect phrasing or non-standard multilingual mixing may fail extraction.
2. **In-Memory Storage**: Current repository implementation is in-memory; persistent PostgreSQL storage is not yet connected.
3. **Mock Embeddings**: Uses a 64-dimensional hash mock rather than a transformer-based embedding model.
4. **Simple Conflict Slots**: Conflicts are resolved on predetermined semantic slots rather than deep world-knowledge ontologies.
5. **No Temporal Expiry**: Plans do not yet automatically expire when their scheduled date passes.

---

## 15. Future Work

- **PostgreSQL + pgvector Integration**: Implement `PgVectorMemoryStore` implementing `BaseMemoryStore` with HNSW vector indices.
- **Sentence Transformers**: Integrate `sentence-transformers/all-MiniLM-L6-v2` for dense semantic embeddings.
- **Memory Consolidation & Decay**: Aggregate repeated memories into composite profile summaries and apply temporal decay to expired plans.
- **Stage 6D Orchestrator**: Integrate Memory Engine retrieval directly into runtime prompt generation alongside Context Engine outputs and Stage 6A inference.
