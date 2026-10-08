"""
Comprehensive Unit Tests for Stage 6D — Conversation Orchestrator.
Validates all requirements A through T:
A. Clear question
B. Invitation
C. Agreement
D. Disagreement
E. Planning
F. High ambiguity
G. Contextual ambiguity
H. Gaming
I. College
J. Technology
K. Closing
L. Memory relevance
M. Irrelevant memory filtering
N. Empty memory store
O. Missing/partial context
P. Persona isolation
Q. Confidence scoring
R. Generation instruction creation
S. All response strategies
T. Deterministic repeated execution
"""

import unittest

from ml.src.context.context_schema import (
    AmbiguityLevel,
    ContextAnalysis,
    ConversationState,
    TopicCategory,
    UserIntent,
)
from ml.src.memory.embedding import DeterministicMockEmbedding
from ml.src.memory.memory_ranker import MemoryRanker, MemoryScoreBreakdown, RankedMemory
from ml.src.memory.memory_retriever import MemoryRetriever
from ml.src.memory.memory_schema import (
    ImportanceLevel,
    Memory,
    MemoryType,
)
from ml.src.memory.memory_store import InMemoryMemoryStore
from ml.src.orchestrator.memory_policy import MemoryPolicy
from ml.src.orchestrator.orchestrator import ConversationOrchestrator
from ml.src.orchestrator.orchestrator_schema import (
    ConversationRequest,
    ResponsePlan,
    ResponseStrategy,
    ResponseTone,
)
from ml.src.orchestrator.response_planner import ResponsePlanner
from ml.src.orchestrator.strategy_selector import StrategySelector


class TestConversationOrchestrator(unittest.TestCase):
    """Unit test suite for Stage 6D Conversation Orchestrator."""

    def setUp(self):
        self.embedding_model = DeterministicMockEmbedding(dimension=64)
        self.memory_store = InMemoryMemoryStore(embedding_model=self.embedding_model)
        self.memory_ranker = MemoryRanker()
        self.memory_retriever = MemoryRetriever(
            store=self.memory_store,
            ranker=self.memory_ranker,
            embedding_model=self.embedding_model,
        )
        self.orchestrator = ConversationOrchestrator(
            memory_retriever=self.memory_retriever
        )

    # =========================================================================
    # A - E: INTENT STRATEGIES
    # =========================================================================

    def test_a_clear_question(self):
        """A. Clear question -> ANSWER strategy."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_1",
            messages=[{"role": "user", "content": "Exam kab se start ho rahe hain?"}],
            context=ContextAnalysis(
                topic=TopicCategory.COLLEGE,
                conversation_state=ConversationState.INQUIRY,
                user_intent=UserIntent.QUESTION,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="Exam kab se start ho rahe hain?",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.response_strategy, ResponseStrategy.ANSWER)
        self.assertIn(plan.tone, {ResponseTone.NEUTRAL, ResponseTone.SERIOUS})

    def test_b_invitation(self):
        """B. Invitation -> ACCEPT strategy."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_2",
            messages=[{"role": "user", "content": "bgmi khelega?"}],
            context=ContextAnalysis(
                topic=TopicCategory.GAMING,
                conversation_state=ConversationState.INQUIRY,
                user_intent=UserIntent.INVITATION,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="bgmi khelega?",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.response_strategy, ResponseStrategy.ACCEPT)
        self.assertEqual(plan.tone, ResponseTone.CASUAL)

    def test_c_agreement(self):
        """C. Agreement -> ACKNOWLEDGE strategy."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_3",
            messages=[{"role": "user", "content": "thik"}],
            context=ContextAnalysis(
                topic=TopicCategory.CASUAL_CHAT,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.AGREEMENT,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="thik",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.response_strategy, ResponseStrategy.ACKNOWLEDGE)

    def test_d_disagreement(self):
        """D. Disagreement -> REACT strategy."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_4",
            messages=[{"role": "user", "content": "nhi bhai aisa nahi hai"}],
            context=ContextAnalysis(
                topic=TopicCategory.CASUAL_CHAT,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.DISAGREEMENT,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="nhi bhai aisa nahi hai",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.response_strategy, ResponseStrategy.REACT)

    def test_e_planning(self):
        """E. Planning -> SUGGEST strategy."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_5",
            messages=[{"role": "user", "content": "kal kab milte hain?"}],
            context=ContextAnalysis(
                topic=TopicCategory.PLANS,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.PLANNING,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="kal kab milte hain?",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.response_strategy, ResponseStrategy.SUGGEST)

    # =========================================================================
    # F - G: AMBIGUITY HANDLING
    # =========================================================================

    def test_f_high_ambiguity(self):
        """F. High ambiguity without context -> ASK_CLARIFICATION."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_6",
            messages=[{"role": "user", "content": "Aaja"}],
            context=ContextAnalysis(
                topic=TopicCategory.CASUAL_CHAT,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.UNKNOWN,
                ambiguity=AmbiguityLevel.HIGH,
                context_depth=1,
                last_user_message="Aaja",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.response_strategy, ResponseStrategy.ASK_CLARIFICATION)

    def test_g_contextual_ambiguity(self):
        """G. Contextual ambiguity resolved by Gaming topic -> ACCEPT."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_7",
            messages=[{"role": "user", "content": "Aaja"}],
            context=ContextAnalysis(
                topic=TopicCategory.GAMING,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.INVITATION,
                ambiguity=AmbiguityLevel.HIGH,
                recent_activity="Invited to play online match",
                context_depth=3,
                last_user_message="Aaja",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.response_strategy, ResponseStrategy.ACCEPT)

    # =========================================================================
    # H - K: DOMAIN TOPICS & STATES
    # =========================================================================

    def test_h_gaming(self):
        """H. Gaming conversation with memory -> ACCEPT and use_memory=True."""
        gaming_mem = Memory(
            id="mem_gaming_1",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Likes gaming",
            topic="gaming",
            importance=ImportanceLevel.HIGH,
        )
        breakdown = MemoryScoreBreakdown(
            final_score=0.75,
            semantic_score=0.8,
            topic_score=1.0,
            importance_score=1.0,
            recency_score=0.9,
            explanation="Test breakdown",
        )
        ranked = RankedMemory(memory=gaming_mem, score=0.75, breakdown=breakdown)

        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_8",
            messages=[{"role": "user", "content": "bgmi khelega?"}],
            context=ContextAnalysis(
                topic=TopicCategory.GAMING,
                conversation_state=ConversationState.INQUIRY,
                user_intent=UserIntent.INVITATION,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="bgmi khelega?",
                speaker_alternation_rate=1.0,
            ),
            memories=[ranked],
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.topic, TopicCategory.GAMING)
        self.assertEqual(plan.response_strategy, ResponseStrategy.ACCEPT)
        self.assertTrue(plan.use_memory)
        self.assertIn("mem_gaming_1", plan.selected_memory_ids)

    def test_i_college(self):
        """I. College notice discussion -> PROVIDE_INFORMATION and SERIOUS tone."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_9",
            messages=[{"role": "user", "content": "Dean ne attendance ka notice nikala hai"}],
            context=ContextAnalysis(
                topic=TopicCategory.COLLEGE,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.INFORMATION,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="Dean ne attendance ka notice nikala hai",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.topic, TopicCategory.COLLEGE)
        self.assertEqual(plan.response_strategy, ResponseStrategy.PROVIDE_INFORMATION)
        self.assertEqual(plan.tone, ResponseTone.SERIOUS)

    def test_j_technology(self):
        """J. Technology issue -> SUGGEST and SUPPORTIVE tone."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_10",
            messages=[{"role": "user", "content": "bhai battery bahut jaldi drain ho rahi"}],
            context=ContextAnalysis(
                topic=TopicCategory.TECHNOLOGY,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.INFORMATION,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="bhai battery bahut jaldi drain ho rahi",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.topic, TopicCategory.TECHNOLOGY)
        self.assertEqual(plan.response_strategy, ResponseStrategy.SUGGEST)
        self.assertEqual(plan.tone, ResponseTone.SUPPORTIVE)

    def test_k_closing(self):
        """K. Closing state -> CLOSE_CONVERSATION."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_11",
            messages=[{"role": "user", "content": "Chal baad me baat karta hu"}],
            context=ContextAnalysis(
                topic=TopicCategory.CASUAL_CHAT,
                conversation_state=ConversationState.CLOSING,
                user_intent=UserIntent.PLANNING,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=2,
                last_user_message="Chal baad me baat karta hu",
                speaker_alternation_rate=1.0,
            ),
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.response_strategy, ResponseStrategy.CLOSE_CONVERSATION)

    # =========================================================================
    # L - N: MEMORY POLICY & EMPTY STORE
    # =========================================================================

    def test_l_memory_relevance(self):
        """L. Memory relevance: Relevant memory selected when score >= threshold."""
        tech_mem = Memory(
            id="mem_tech_1",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Prefers Java",
            topic="technology",
            importance=ImportanceLevel.HIGH,
        )
        ranked = RankedMemory(
            memory=tech_mem,
            score=0.65,
            breakdown=MemoryScoreBreakdown(
                final_score=0.65,
                semantic_score=0.7,
                topic_score=1.0,
                importance_score=1.0,
                recency_score=0.9,
                explanation="Tech match",
            ),
        )
        use_mem, selected_ids, _ = MemoryPolicy.evaluate_memories(
            candidate_memories=[ranked],
            current_topic=TopicCategory.TECHNOLOGY,
        )
        self.assertTrue(use_mem)
        self.assertEqual(selected_ids, ["mem_tech_1"])

    def test_m_irrelevant_memory_filtering(self):
        """M. Irrelevant memory filtering: Cross-domain memory rejected."""
        college_mem = Memory(
            id="mem_college_1",
            persona_id="vivek",
            memory_type=MemoryType.FACT,
            content="Studies B.Tech CSE",
            topic="college",
            importance=ImportanceLevel.HIGH,
        )
        ranked = RankedMemory(
            memory=college_mem,
            score=0.60,
            breakdown=MemoryScoreBreakdown(
                final_score=0.60,
                semantic_score=0.4,
                topic_score=0.0,
                importance_score=1.0,
                recency_score=0.9,
                explanation="College mem",
            ),
        )
        # In a GAMING conversation, college memory should be filtered out
        use_mem, selected_ids, _ = MemoryPolicy.evaluate_memories(
            candidate_memories=[ranked],
            current_topic=TopicCategory.GAMING,
        )
        self.assertFalse(use_mem)
        self.assertEqual(selected_ids, [])

    def test_n_empty_memory_store(self):
        """N. Empty memory store: Safely produces plan with use_memory=False."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_12",
            messages=[{"role": "user", "content": "Kaisa hai bhai?"}],
            context=ContextAnalysis(
                topic=TopicCategory.CASUAL_CHAT,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.QUESTION,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=1,
                last_user_message="Kaisa hai bhai?",
                speaker_alternation_rate=1.0,
            ),
            memories=[],
        )
        plan = self.orchestrator.plan(req)
        self.assertFalse(plan.use_memory)
        self.assertEqual(plan.selected_memory_ids, [])
        self.assertIsNotNone(plan.generation_instruction)

    # =========================================================================
    # O - P: FALLBACK & PERSONA ISOLATION
    # =========================================================================

    def test_o_missing_partial_context(self):
        """O. Missing context: Orchestrator falls back cleanly using ContextAnalyzer."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_13",
            messages=[{"role": "user", "content": "bgmi khelega?"}],
            context=None,
        )
        plan = self.orchestrator.plan(req)
        self.assertEqual(plan.topic, TopicCategory.GAMING)
        self.assertEqual(plan.response_strategy, ResponseStrategy.ACCEPT)

    def test_p_persona_isolation(self):
        """P. Persona isolation: Memories for Alice never queried for Bob."""
        self.memory_store.save(
            Memory(
                id="mem_alice",
                persona_id="persona_alice",
                memory_type=MemoryType.PREFERENCE,
                content="Likes gaming",
                topic="gaming",
            )
        )
        # Query as Bob with unpopulated memories field
        req_bob = ConversationRequest(
            persona_id="persona_bob",
            conversation_id="conv_bob",
            messages=[{"role": "user", "content": "khelega?"}],
        )
        plan_bob = self.orchestrator.plan(req_bob)
        self.assertNotIn("mem_alice", plan_bob.selected_memory_ids)

    # =========================================================================
    # Q - T: SCORING, INSTRUCTIONS, TAXONOMY, DETERMINISM
    # =========================================================================

    def test_q_confidence_scoring(self):
        """Q. Confidence scoring: Clear intent + low ambiguity > ambiguous turn."""
        conf_high = ResponsePlanner.calculate_confidence(
            user_intent=UserIntent.QUESTION,
            ambiguity=AmbiguityLevel.LOW,
            context_depth=3,
            response_strategy=ResponseStrategy.ANSWER,
            use_memory=True,
        )
        conf_low = ResponsePlanner.calculate_confidence(
            user_intent=UserIntent.UNKNOWN,
            ambiguity=AmbiguityLevel.HIGH,
            context_depth=1,
            response_strategy=ResponseStrategy.ASK_CLARIFICATION,
            use_memory=False,
        )
        self.assertGreater(conf_high, conf_low)
        self.assertGreaterEqual(conf_high, 0.70)
        self.assertLessEqual(conf_low, 0.50)

    def test_r_generation_instruction_creation(self):
        """R. Generation instruction creation: Contains strategy, tone, topic, and context."""
        instruction = ResponsePlanner.generate_instruction(
            response_strategy=ResponseStrategy.ACCEPT,
            tone=ResponseTone.CASUAL,
            topic=TopicCategory.GAMING,
            user_intent=UserIntent.INVITATION,
            use_memory=True,
        )
        self.assertIn("casual", instruction)
        self.assertIn("gaming", instruction)
        self.assertIn("natural", instruction)

    def test_s_all_response_strategies(self):
        """S. All 10 response strategies in taxonomy are valid enum members."""
        expected_strategies = {
            "answer",
            "acknowledge",
            "ask_clarification",
            "accept",
            "decline",
            "suggest",
            "react",
            "continue_banter",
            "provide_information",
            "close_conversation",
        }
        actual_strategies = {s.value for s in ResponseStrategy}
        self.assertEqual(expected_strategies, actual_strategies)

    def test_t_deterministic_repeated_execution(self):
        """T. Deterministic repeated execution: Identical inputs yield identical outputs 50 times."""
        req = ConversationRequest(
            persona_id="vivek",
            conversation_id="conv_det",
            messages=[{"role": "user", "content": "bgmi khelega?"}],
            context=ContextAnalysis(
                topic=TopicCategory.GAMING,
                conversation_state=ConversationState.INQUIRY,
                user_intent=UserIntent.INVITATION,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=2,
                last_user_message="bgmi khelega?",
                speaker_alternation_rate=1.0,
            ),
        )
        baseline_plan = self.orchestrator.plan(req).to_dict()
        for _ in range(50):
            repeated_plan = self.orchestrator.plan(req).to_dict()
            self.assertEqual(baseline_plan, repeated_plan)


if __name__ == "__main__":
    unittest.main()
