# Stage 6D — Conversation Orchestrator Report

## 1. Architecture

Stage 6D introduces the **Conversation Orchestrator**, the decision-making and coordination layer for the Persona Engine. The Orchestrator sits between dialogue input, the Context Engine (Stage 6B), the Memory Engine (Stage 6C), and the downstream Model Gateway / Inference Service (Stage 6A).

Its core responsibility is to answer:
> **"What should the system do next?"**

The Orchestrator produces a structured, strongly-typed `ResponsePlan` guiding downstream text generation. It does **not** call the persona model or generate final natural-language responses itself.

```
User Message
     │
     ▼
[Conversation Orchestrator]
     ├── Context Engine (Stage 6B)
     ├── Memory Engine (Stage 6C)
     ├── Strategy Selector (Deterministic Rules)
     ├── Memory Policy (Quality & Topic Filter)
     └── Response Planner (Tone, Confidence, Prompt Guidance)
     │
     ▼
[ResponsePlan] (Structured Guidance)
     │
     ▼
[Model Gateway / Inference Service] (Stage 6A)
     │
     ▼
[Stage 5 Persona LoRA Model]
     │
     ▼
Final Natural-Language Persona Response
```

---

## 2. Files Created

1. `ml/src/orchestrator/orchestrator_schema.py` — Strongly typed Pydantic models (`ConversationRequest`, `ResponsePlan`, `ResponseStrategy`, `ResponseTone`).
2. `ml/src/orchestrator/strategy_selector.py` — Deterministic, rule-based strategy selector mapping conversational dimensions to dialogue action strategies.
3. `ml/src/orchestrator/memory_policy.py` — Relevance gating, domain topic filtering, and memory safety policies.
4. `ml/src/orchestrator/response_planner.py` — Synthesizes strategy, tone, memory decisions, confidence scores, and generation instructions into `ResponsePlan`.
5. `ml/src/orchestrator/orchestrator.py` — Central `ConversationOrchestrator` coordinating context intelligence, memory retrieval, and planning.
6. `ml/src/orchestrator/__init__.py` — Clean public package exports.
7. `ml/src/orchestrator/README.md` — Architectural documentation and responsibility division table.
8. `ml/tests/test_orchestrator.py` — Unit test suite covering Scenarios A through T (20 unit tests).
9. `ml/scripts/test_orchestrator.py` — Integration benchmark running 10 realistic dialogue scenarios.
10. `ml/data/stage5/reports/stage6d_orchestrator_report.md` — This milestone report.

**No baseline files from Stages 5, 6A, 6B, or 6C were modified.**

---

## 3. Input Schema

`ConversationRequest` encapsulates the necessary dialogue context and precomputed upstream signals:
- `persona_id: str` (default: `"vivek"`)
- `conversation_id: str` (default: `"conv_default"`)
- `messages: List[Dict[str, str]]` — Dialogue history.
- `context: Optional[ContextAnalysis]` — Upstream context analysis from Stage 6B.
- `memories: Optional[List[Union[RankedMemory, Memory]]]` — Retrieved candidate memories from Stage 6C.
- `previous_response: Optional[str]` — Prior model response.
- `persona_metadata: Optional[Dict[str, Any]]` — Persona settings.

---

## 4. Output Schema

`ResponsePlan` provides structured guidance for Stage 6A:
- `topic: TopicCategory`
- `user_intent: UserIntent`
- `conversation_state: ConversationState`
- `response_strategy: ResponseStrategy`
- `tone: ResponseTone`
- `use_memory: bool`
- `selected_memory_ids: List[str]`
- `selected_memories: List[Dict[str, Any]]`
- `context_depth: int`
- `ambiguity: AmbiguityLevel`
- `generation_instruction: str`
- `confidence: float` ($0.10 - 1.00$)

---

## 5. Strategy Taxonomy

The controlled `ResponseStrategy` taxonomy consists of:
- `answer`: Directly answer queries or factual questions.
- `acknowledge`: Confirm or acknowledge user remarks (`"thik"`, `"sahi hai"`).
- `ask_clarification`: Inquire for disambiguation on ambiguous turns (`"Aaja"` without context).
- `accept`: Agree to gaming, social, or planning invitations (`"khelega?"`).
- `decline`: Politely decline an invitation.
- `suggest`: Offer advice, troubleshooting tips, or proposals.
- `react`: Express an authentic emotional reaction.
- `continue_banter`: Engage in playful back-and-forth retorts.
- `provide_information`: Relay administrative or factual updates (notices, dates).
- `close_conversation`: Gracefully wrap up and conclude dialogue.

---

## 6. Strategy Selection

Implemented in `StrategySelector`:
- **Closing Signals**: Closing tokens (`"bye"`, `"baad me baat"`, `"chalta hu"`) or state `CLOSING` trigger `CLOSE_CONVERSATION`.
- **Questions**: Intent `QUESTION` triggers `ANSWER`.
- **Domain Issues / Notices**:
  - Technology problem indicators (`"battery drain"`, `"crash"`, `"issue"`) trigger `SUGGEST`.
  - College administrative notices (`"notice"`, `"attendance"`, `"dean"`) trigger `PROVIDE_INFORMATION`.
- **Invitations**: Intent `INVITATION` triggers `ACCEPT`.
- **High Ambiguity**:
  - Unresolvable isolated turns (`"Aaja"`) trigger `ASK_CLARIFICATION`.
  - Contextually grounded turns (gaming context) trigger `ACCEPT`.
- **Agreements / Disagreements**:
  - Quick agreement (`"thik"`) triggers `ACKNOWLEDGE`.
  - Disagreement (`"nhi bhai"`) triggers `REACT` or `CONTINUE_BANTER`.

---

## 7. Memory Policy

Implemented in `MemoryPolicy`:
- **Score Threshold**: Candidate memories require a composite ranking score $\ge 0.45$.
- **Topic Alignment**: Memories must match the conversation's active domain (e.g. gaming memories in gaming chats). Cross-domain memories are filtered out.
- **Volume Cap**: Maximum 2 memories are passed to prompt context.
- **Closing Suppression**: When closing a dialogue, memories are suppressed (`use_memory = False`).
- **Safety**: Memories are never forced into the response; they serve as optional background awareness.

---

## 8. Tone Selection

Controlled via `ResponseTone`:
- `SERIOUS`: College administrative notices, exam warnings, attendance policies.
- `SUPPORTIVE`: Technology problems, battery drain, debugging issues.
- `CASUAL`: Standard gaming, movies, social chat, and dialogue wrap-ups.
- `HUMOROUS` / `TEASING`: Playful banter exchanges.
- `NEUTRAL`: General informational exchanges.

---

## 9. Confidence

Confidence is deterministically calibrated using 5 additive factors:
$$\text{Confidence} = C_{\text{intent}} + C_{\text{ambiguity}} + C_{\text{depth}} + C_{\text{strategy}} + C_{\text{memory}}$$

- Intent Clarity: Clear intents $= 0.25$; generic $= 0.15$.
- Ambiguity: Low $= 0.35$; Medium $= 0.20$; High $= 0.05$.
- Context Depth: $\ge 3$ turns $= 0.15$; 2 turns $= 0.10$; 1 turn $= 0.05$.
- Strategy Certainty: High certainty actions $= 0.15$; clarifications $= 0.10$.
- Memory Support: Relevant memory used $= 0.10$; baseline $= 0.05$.

Total confidence is clamped between $0.10$ and $1.00$.

---

## 10. Fallback Behavior

- **Empty Messages**: Fallback plan with `strategy=REACT`, `tone=CASUAL`, `confidence=0.30`.
- **Missing Context**: Automatically invokes `ContextAnalyzer().analyze(messages)`.
- **Empty Memories**: Safely sets `use_memory=False`, `selected_memory_ids=[]`.
- The system never crashes or raises unhandled exceptions on incomplete inputs.

---

## 11. Generation Instruction

Generates actionable guidance for downstream text generation:
- *Example for gaming invitation:*
  `"Respond with a casual acceptance to the gaming invitation. Keep it natural and concise. Incorporate relevant context naturally without explicitly reciting memory records."`
- *Example for college notice:*
  `"Share relevant, serious information regarding the college update. Incorporate relevant context naturally without explicitly reciting memory records."`

---

## 12. Determinism

The Orchestrator contains zero random elements and zero model calls. Identical inputs yield identical `ResponsePlan` outputs across 50 consecutive runs.

---

## 13. Unit Test Results (`ml/tests/test_orchestrator.py`)

All 20 unit tests passed:
- `test_a_clear_question` — PASS (`strategy = ANSWER`)
- `test_b_invitation` — PASS (`strategy = ACCEPT`)
- `test_c_agreement` — PASS (`strategy = ACKNOWLEDGE`)
- `test_d_disagreement` — PASS (`strategy = REACT`)
- `test_e_planning` — PASS (`strategy = SUGGEST`)
- `test_f_high_ambiguity` — PASS (`strategy = ASK_CLARIFICATION`)
- `test_g_contextual_ambiguity` — PASS (`strategy = ACCEPT`)
- `test_h_gaming` — PASS (`topic = GAMING`, `use_memory = True`)
- `test_i_college` — PASS (`topic = COLLEGE`, `strategy = PROVIDE_INFORMATION`, `tone = SERIOUS`)
- `test_j_technology` — PASS (`topic = TECHNOLOGY`, `strategy = SUGGEST`, `tone = SUPPORTIVE`)
- `test_k_closing` — PASS (`strategy = CLOSE_CONVERSATION`)
- `test_l_memory_relevance` — PASS (Memory selected when score $\ge 0.45$)
- `test_m_irrelevant_memory_filtering` — PASS (Cross-domain memory rejected)
- `test_n_empty_memory_store` — PASS (Graceful `use_memory = False`)
- `test_o_missing_partial_context` — PASS (Fallback via `ContextAnalyzer`)
- `test_p_persona_isolation` — PASS (Multi-tenant persona isolation)
- `test_q_confidence_scoring` — PASS (Clear intent > ambiguous intent)
- `test_r_generation_instruction_creation` — PASS (Guidance string assembled)
- `test_s_all_response_strategies` — PASS (All 10 enum strategies valid)
- `test_t_deterministic_repeated_execution` — PASS (50/50 runs identical)

---

## 14. Integration Benchmark (`ml/scripts/test_orchestrator.py`)

Executed 10 realistic conversation flows:
1. **Gaming Invitation** (`"bhai bgmi khelega?"`): `strategy=accept`, `tone=casual`, `confidence=0.75` **[PASS]**
2. **College Discussion** (`"Dean ne attendance ka notice nikala hai"`): `strategy=provide_information`, `tone=serious`, `confidence=0.80` **[PASS]**
3. **Technology Problem** (`"bhai battery bahut jaldi drain ho rahi"`): `strategy=suggest`, `tone=supportive`, `confidence=0.75` **[PASS]**
4. **Movie Conversation** (`"Marvel ki new movie dekhi kya?"`): `strategy=answer`, `tone=casual`, `confidence=0.90` **[PASS]**
5. **Ambiguous 'Aaja'** (Isolated): `strategy=ask_clarification`, `ambiguity=high`, `confidence=0.50` **[PASS]**
6. **Contextual 'Aaja'** (Gaming context): `strategy=accept`, `topic=gaming`, `confidence=0.70` **[PASS]**
7. **Direct 'Khelega?'**: `strategy=accept`, `topic=gaming`, `confidence=0.60` **[PASS]**
8. **Quick Agreement 'Thik'**: `strategy=acknowledge`, `confidence=0.60` **[PASS]**
9. **Decline/Reaction 'Nhi bhai'**: `strategy=react`, `confidence=0.55` **[PASS]**
10. **Closing Conversation** (`"Chal baad me baat karta hu"`): `strategy=close_conversation`, `use_memory=False`, `confidence=0.90` **[PASS]**

**Result: 10/10 benchmarks passed.**

---

## 15. Full Regression

```
Ran 128 tests in 1.850s
OK (128/128 passing, 0 failures, 0 errors)
```
- Baseline tests: 108 passed
- Orchestrator tests: 20 passed

---

## 16. Limitations

1. **No Direct Text Generation**: As per architectural design, the Orchestrator does not generate natural language text.
2. **Rule-Based Mapping**: Edge-case multilingual idioms not covered by rules fall back to `REACT`.
3. **No External Agent Orchestration**: Multi-agent task delegation and external tool execution are out of scope.

---

## 17. Future Work

- **Stage 6E / Prompt Assembly Integration**: Inject `ResponsePlan` parameters into dynamic system prompt templates before invoking Stage 6A `/generate`.
- **Spring Boot API Gateway**: Route incoming mobile chat requests to Context Engine $\rightarrow$ Memory Engine $\rightarrow$ Orchestrator $\rightarrow$ Model Gateway.
