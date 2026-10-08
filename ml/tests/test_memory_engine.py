"""
Comprehensive Unit Tests for Stage 6C — Memory Engine.
Validates all requirements A through Q:
A. Fact extraction
B. Preference extraction
C. Goal extraction
D. Plan extraction
E. Relationship extraction
F. Experience extraction
G. Non-memory rejection
H. Non-memory invitation rejection
I. Duplicate detection & normalization
J. Conflict handling & supersession
K. Importance scoring
L. Retrieval relevance
M. Importance ranking
N. Recency ranking
O. Persona isolation
P. Empty memory store safety
Q. Context Engine integration (ContextAnalysis consumption)
"""

from datetime import datetime, timedelta, timezone
import unittest

from ml.src.context.context_schema import (
    AmbiguityLevel,
    ContextAnalysis,
    ConversationState,
    TopicCategory,
    UserIntent,
)
from ml.src.memory.embedding import (
    DeterministicMockEmbedding,
    cosine_similarity,
)
from ml.src.memory.memory_extractor import MemoryExtractor
from ml.src.memory.memory_ranker import MemoryRanker
from ml.src.memory.memory_retriever import MemoryRetriever
from ml.src.memory.memory_schema import (
    ImportanceLevel,
    Memory,
    MemoryStatus,
    MemoryType,
)
from ml.src.memory.memory_store import (
    InMemoryMemoryStore,
    normalize_content_text,
)


class TestMemoryEngine(unittest.TestCase):
    """Unit test suite for Persona Engine Memory Layer."""

    def setUp(self):
        self.embedding_model = DeterministicMockEmbedding(dimension=64)
        self.store = InMemoryMemoryStore(embedding_model=self.embedding_model)
        self.ranker = MemoryRanker()
        self.retriever = MemoryRetriever(
            store=self.store,
            ranker=self.ranker,
            embedding_model=self.embedding_model,
        )

    # =========================================================================
    # A - F: TAXONOMY & EXTRACTION TESTS
    # =========================================================================

    def test_a_fact_extraction(self):
        """A. Fact extraction: 'Main BTech CSE kar raha hu.' -> FACT"""
        utterance = "Main BTech CSE kar raha hu."
        memories = MemoryExtractor.extract_from_text(utterance, persona_id="user_1")
        self.assertEqual(len(memories), 1)
        mem = memories[0]
        self.assertEqual(mem.memory_type, MemoryType.FACT)
        self.assertEqual(mem.content, "Studies B.Tech CSE")
        self.assertEqual(mem.topic, "college")

    def test_b_preference_extraction(self):
        """B. Preference extraction: 'Mujhe Java pasand hai.' -> PREFERENCE"""
        utterance = "Mujhe Java pasand hai."
        memories = MemoryExtractor.extract_from_text(utterance, persona_id="user_1")
        self.assertEqual(len(memories), 1)
        mem = memories[0]
        self.assertEqual(mem.memory_type, MemoryType.PREFERENCE)
        self.assertEqual(mem.content, "Prefers Java")
        self.assertEqual(mem.importance, ImportanceLevel.HIGH)

    def test_c_goal_extraction(self):
        """C. Goal extraction: 'Main backend developer banna chahta hu.' -> GOAL"""
        utterance = "Main backend developer banna chahta hu."
        memories = MemoryExtractor.extract_from_text(utterance, persona_id="user_1")
        self.assertEqual(len(memories), 1)
        mem = memories[0]
        self.assertEqual(mem.memory_type, MemoryType.GOAL)
        self.assertEqual(mem.content, "Wants to become a backend developer")
        self.assertEqual(mem.importance, ImportanceLevel.HIGH)

    def test_d_plan_extraction(self):
        """D. Plan extraction: 'Kal interview hai.' -> PLAN"""
        utterance = "Kal interview hai."
        memories = MemoryExtractor.extract_from_text(utterance, persona_id="user_1")
        self.assertEqual(len(memories), 1)
        mem = memories[0]
        self.assertEqual(mem.memory_type, MemoryType.PLAN)
        self.assertEqual(mem.content, "Has an upcoming interview")
        self.assertEqual(mem.importance, ImportanceLevel.LOW)

    def test_e_relationship_extraction(self):
        """E. Relationship extraction: 'Rahul mera college friend hai.' -> RELATIONSHIP"""
        utterance = "Rahul mera college friend hai."
        memories = MemoryExtractor.extract_from_text(utterance, persona_id="user_1")
        self.assertEqual(len(memories), 1)
        mem = memories[0]
        self.assertEqual(mem.memory_type, MemoryType.RELATIONSHIP)
        self.assertEqual(mem.content, "Rahul is a college friend")

    def test_f_experience_extraction(self):
        """F. Experience extraction: 'Maine SIH me participate kiya tha.' -> EXPERIENCE"""
        utterance = "Maine SIH me participate kiya tha."
        memories = MemoryExtractor.extract_from_text(utterance, persona_id="user_1")
        self.assertEqual(len(memories), 1)
        mem = memories[0]
        self.assertEqual(mem.memory_type, MemoryType.EXPERIENCE)
        self.assertEqual(mem.content, "Participated in Smart India Hackathon (SIH)")

    # =========================================================================
    # G - H: NON-MEMORY REJECTION
    # =========================================================================

    def test_g_non_memory_message(self):
        """G. Non-memory message: 'haan bhai' -> zero memories"""
        for ephemeral in ["haan bhai", "thik hai", "ok", "yep", "arre yaar"]:
            memories = MemoryExtractor.extract_from_text(ephemeral)
            self.assertEqual(len(memories), 0, f"Expected 0 memories for ephemeral turn: {ephemeral}")

    def test_h_non_memory_invitation(self):
        """H. Non-memory invitation: 'Aaja game khelte hain.' -> zero memories"""
        invitations = [
            "Aaja game khelte hain.",
            "khelega?",
            "kal milte hain",
            "canteen chale?",
        ]
        for inv in invitations:
            memories = MemoryExtractor.extract_from_text(inv)
            self.assertEqual(len(memories), 0, f"Expected 0 memories for invitation: {inv}")

    # =========================================================================
    # I: DUPLICATE DETECTION & NORMALIZATION
    # =========================================================================

    def test_i_duplicate_detection(self):
        """I. Duplicate detection: Identical or normalized facts merged without duplicate row."""
        first_memories = MemoryExtractor.extract_from_text("Mujhe Java pasand hai.", persona_id="user_1")
        self.assertEqual(len(first_memories), 1)
        saved_1, action_1 = self.store.save_or_merge(first_memories[0])
        self.assertEqual(action_1, "created")
        self.assertEqual(self.store.count("user_1"), 1)

        # Second utterance representing same fact
        second_memories = MemoryExtractor.extract_from_text("Mujhe Java bahut pasand hai.", persona_id="user_1")
        self.assertEqual(len(second_memories), 1)
        saved_2, action_2 = self.store.save_or_merge(second_memories[0])
        self.assertEqual(action_2, "merged")
        self.assertEqual(saved_2.id, saved_1.id)
        self.assertEqual(self.store.count("user_1"), 1)

        # Test normalization text helper directly
        self.assertEqual(
            normalize_content_text("I really like Java."),
            normalize_content_text("Likes Java"),
        )

    # =========================================================================
    # J: CONFLICT HANDLING & SUPERSEDING
    # =========================================================================

    def test_j_conflict_handling(self):
        """J. Conflict handling: 'Prefers Java' superseded by 'Prefers Python'."""
        mem_java = Memory(
            persona_id="user_1",
            memory_type=MemoryType.PREFERENCE,
            content="Prefers Java",
            importance=ImportanceLevel.HIGH,
            topic="technology",
        )
        saved_java, action_java = self.store.save_or_merge(mem_java)
        self.assertEqual(action_java, "created")
        self.assertEqual(saved_java.status, MemoryStatus.ACTIVE)

        # Conflicting preference
        mem_python = Memory(
            persona_id="user_1",
            memory_type=MemoryType.PREFERENCE,
            content="Prefers Python",
            importance=ImportanceLevel.HIGH,
            topic="technology",
        )
        saved_python, action_python = self.store.save_or_merge(mem_python)
        self.assertEqual(action_python, "conflict_resolved")

        # Verify old memory is marked SUPERSEDED with pointer
        old_mem = self.store.get(saved_java.id)
        self.assertIsNotNone(old_mem)
        self.assertEqual(old_mem.status, MemoryStatus.SUPERSEDED)
        self.assertEqual(old_mem.superseded_by, saved_python.id)

        # Active list contains only the new memory
        active_list = self.store.list_by_persona("user_1", status=MemoryStatus.ACTIVE)
        self.assertEqual(len(active_list), 1)
        self.assertEqual(active_list[0].content, "Prefers Python")

    # =========================================================================
    # K: IMPORTANCE SCORING
    # =========================================================================

    def test_k_importance_scoring(self):
        """K. Importance scoring rules: LOW vs MEDIUM vs HIGH."""
        plan_mem = MemoryExtractor.extract_from_text("Kal interview hai.")[0]
        self.assertEqual(plan_mem.importance, ImportanceLevel.LOW)

        exp_mem = MemoryExtractor.extract_from_text("Maine SIH me participate kiya tha.")[0]
        self.assertEqual(exp_mem.importance, ImportanceLevel.MEDIUM)

        goal_mem = MemoryExtractor.extract_from_text("Main backend developer banna chahta hu.")[0]
        self.assertEqual(goal_mem.importance, ImportanceLevel.HIGH)

    # =========================================================================
    # L: RETRIEVAL RELEVANCE
    # =========================================================================

    def test_l_retrieval_relevance(self):
        """L. Retrieval relevance: Gaming query ranks gaming memory above college & movies."""
        mem_gaming = Memory(
            persona_id="user_1",
            memory_type=MemoryType.PREFERENCE,
            content="Likes gaming",
            topic="gaming",
            importance=ImportanceLevel.HIGH,
        )
        mem_college = Memory(
            persona_id="user_1",
            memory_type=MemoryType.FACT,
            content="Studies B.Tech CSE",
            topic="college",
            importance=ImportanceLevel.HIGH,
        )
        mem_movies = Memory(
            persona_id="user_1",
            memory_type=MemoryType.PREFERENCE,
            content="Likes Marvel movies",
            topic="movies",
            importance=ImportanceLevel.HIGH,
        )
        self.store.save(mem_gaming)
        self.store.save(mem_college)
        self.store.save(mem_movies)

        retrieved = self.retriever.retrieve(
            persona_id="user_1",
            query="bhai bgmi khelega?",
            top_k=3,
        )
        self.assertGreater(len(retrieved), 0)
        self.assertEqual(retrieved[0].memory.topic, "gaming")
        self.assertEqual(retrieved[0].memory.content, "Likes gaming")

    # =========================================================================
    # M: IMPORTANCE RANKING
    # =========================================================================

    def test_m_importance_ranking(self):
        """M. Importance ranking: Higher importance memory ranks higher ceteris paribus."""
        mem_high = Memory(
            persona_id="user_1",
            memory_type=MemoryType.GOAL,
            content="Wants to become a backend developer",
            topic="technology",
            importance=ImportanceLevel.HIGH,
        )
        mem_low = Memory(
            persona_id="user_1",
            memory_type=MemoryType.PLAN,
            content="Has an upcoming interview",
            topic="technology",
            importance=ImportanceLevel.LOW,
        )
        self.store.save(mem_high)
        self.store.save(mem_low)

        retrieved = self.retriever.retrieve(
            persona_id="user_1",
            query="technology developer career",
            top_k=2,
        )
        self.assertEqual(retrieved[0].memory.importance, ImportanceLevel.HIGH)
        self.assertGreater(retrieved[0].score, retrieved[1].score)

    # =========================================================================
    # N: RECENCY RANKING
    # =========================================================================

    def test_n_recency_ranking(self):
        """N. Recency ranking: More recent memory scores higher than aged memory."""
        now = datetime.now(timezone.utc)
        thirty_days_ago = (now - timedelta(days=30)).isoformat()

        old_mem = Memory(
            persona_id="user_1",
            memory_type=MemoryType.EXPERIENCE,
            content="Completed software project",
            topic="technology",
            importance=ImportanceLevel.MEDIUM,
            created_at=thirty_days_ago,
            updated_at=thirty_days_ago,
        )
        fresh_mem = Memory(
            persona_id="user_1",
            memory_type=MemoryType.EXPERIENCE,
            content="Completed software project",
            topic="technology",
            importance=ImportanceLevel.MEDIUM,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
        )
        self.store.save(old_mem)
        self.store.save(fresh_mem)

        retrieved = self.retriever.retrieve(
            persona_id="user_1",
            query="development project",
            top_k=2,
        )
        self.assertEqual(retrieved[0].memory.id, fresh_mem.id)
        self.assertGreater(
            retrieved[0].breakdown.recency_score,
            retrieved[1].breakdown.recency_score,
        )

    # =========================================================================
    # O: PERSONA ISOLATION
    # =========================================================================

    def test_o_persona_isolation(self):
        """O. Persona isolation: Memories belonging to persona A never returned for persona B."""
        mem_alice = Memory(
            persona_id="persona_alice",
            memory_type=MemoryType.PREFERENCE,
            content="Likes gaming",
            topic="gaming",
        )
        self.store.save(mem_alice)

        # Query as Bob
        results_bob = self.retriever.retrieve(
            persona_id="persona_bob",
            query="gaming",
        )
        self.assertEqual(len(results_bob), 0)

        # Query as Alice
        results_alice = self.retriever.retrieve(
            persona_id="persona_alice",
            query="gaming",
        )
        self.assertEqual(len(results_alice), 1)
        self.assertEqual(results_alice[0].memory.persona_id, "persona_alice")

    # =========================================================================
    # P: EMPTY MEMORY STORE SAFETY
    # =========================================================================

    def test_p_empty_memory_store(self):
        """P. Empty memory store: Safely returns empty list without exception."""
        empty_store = InMemoryMemoryStore()
        retriever = MemoryRetriever(store=empty_store)
        results = retriever.retrieve(persona_id="user_empty", query="any query")
        self.assertEqual(results, [])

    # =========================================================================
    # Q: CONTEXT ENGINE INTEGRATION
    # =========================================================================

    def test_q_context_integration(self):
        """Q. Context integration: ContextAnalysis topic=gaming boosts gaming memories."""
        mem_gaming = Memory(
            persona_id="user_1",
            memory_type=MemoryType.PREFERENCE,
            content="Likes gaming",
            topic="gaming",
            importance=ImportanceLevel.HIGH,
        )
        mem_college = Memory(
            persona_id="user_1",
            memory_type=MemoryType.FACT,
            content="Studies B.Tech CSE",
            topic="college",
            importance=ImportanceLevel.HIGH,
        )
        self.store.save(mem_gaming)
        self.store.save(mem_college)

        # Ambiguous message "Aaja" with Gaming context
        context_analysis = ContextAnalysis(
            topic=TopicCategory.GAMING,
            conversation_state=ConversationState.ONGOING,
            user_intent=UserIntent.INVITATION,
            ambiguity=AmbiguityLevel.HIGH,
            recent_activity="Discussed gaming session",
            context_depth=3,
            last_user_message="Aaja",
            speaker_alternation_rate=1.0,
        )

        ranked = self.retriever.retrieve(
            persona_id="user_1",
            context_analysis=context_analysis,
            top_k=2,
        )
        self.assertGreater(len(ranked), 0)
        self.assertEqual(ranked[0].memory.topic, "gaming")
        self.assertEqual(ranked[0].breakdown.topic_score, 1.0)

    # =========================================================================
    # ADDITIONAL TESTS: CRUD, ACCESS TIME, CONVERSATION, SERIALIZATION
    # =========================================================================

    def test_store_crud_and_access(self):
        """Validates CRUD operations and access timestamp updates."""
        mem = Memory(
            persona_id="user_crud",
            memory_type=MemoryType.FACT,
            content="Lives in Delhi",
            topic="casual_chat",
        )
        saved = self.store.save(mem)
        self.assertEqual(self.store.get(saved.id).content, "Lives in Delhi")
        self.assertEqual(self.store.count("user_crud"), 1)

        # Update last accessed
        self.assertIsNone(saved.last_accessed_at)
        updated_acc = self.store.update_last_accessed(saved.id)
        self.assertIsNotNone(updated_acc.last_accessed_at)

        # Update content
        saved.content = "Lives in South Delhi"
        self.store.update(saved)
        self.assertEqual(self.store.get(saved.id).content, "Lives in South Delhi")

        # Delete
        deleted = self.store.delete(saved.id)
        self.assertTrue(deleted)
        self.assertIsNone(self.store.get(saved.id))
        self.assertEqual(self.store.count("user_crud"), 0)

    def test_extract_from_conversation(self):
        """Extracts memories from multi-turn dialogue focusing on user utterances."""
        conversation = [
            {"role": "user", "content": "Kaisa hai bhai?"},
            {"role": "assistant", "content": "Badhiya bhai, tu bata!"},
            {"role": "user", "content": "Bas chal raha hai. Main BTech CSE kar raha hu."},
            {"role": "assistant", "content": "Sahi hai! Kaunse college se?"},
            {"role": "user", "content": "Mera college Noida me hai."},
            {"role": "assistant", "content": "Nice location."},
            {"role": "user", "content": "haan bhai"},
        ]
        extracted = MemoryExtractor.extract_from_conversation(
            messages=conversation,
            persona_id="user_dialogue",
            source_conversation_id="conv_101",
        )
        self.assertEqual(len(extracted), 2)
        contents = {m.content for m in extracted}
        self.assertIn("Studies B.Tech CSE", contents)
        self.assertIn("College is located in Noida", contents)

    def test_memory_serialization(self):
        """Tests JSON dictionary serialization and reconstruction."""
        mem = Memory(
            persona_id="user_serial",
            memory_type=MemoryType.PREFERENCE,
            content="Likes Marvel movies",
            importance=ImportanceLevel.HIGH,
            topic="movies",
        )
        d = mem.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["content"], "Likes Marvel movies")
        self.assertEqual(d["memory_type"], "preference")
        self.assertEqual(d["importance"], "high")

        reconstructed = Memory(**d)
        self.assertEqual(reconstructed.id, mem.id)
        self.assertEqual(reconstructed.content, mem.content)


if __name__ == "__main__":
    unittest.main()

