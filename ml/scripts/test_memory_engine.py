"""
Integration Benchmark for Persona Engine Persistent Memory Layer (Stage 6C).
Executes realistic conversation scenarios across 5 benchmarks:
- Benchmark 1: Gaming retrieval ('bhai bgmi khelega?')
- Benchmark 2: College retrieval ('Dean ne attendance ka notice nikala hai')
- Benchmark 3: Technology retrieval ('bhai battery bahut jaldi drain ho rahi')
- Benchmark 4: Movie preferences retrieval ('Marvel ki new movie dekhi?')
- Benchmark 5: Contextual disambiguation for ambiguous turns ('Aaja' in gaming vs canteen context)
"""

import sys
from pathlib import Path
from typing import List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.context.context_schema import (
    AmbiguityLevel,
    ContextAnalysis,
    ConversationState,
    TopicCategory,
    UserIntent,
)
from ml.src.memory.embedding import DeterministicMockEmbedding
from ml.src.memory.memory_ranker import MemoryRanker
from ml.src.memory.memory_retriever import MemoryRetriever
from ml.src.memory.memory_schema import (
    ImportanceLevel,
    Memory,
    MemoryType,
)
from ml.src.memory.memory_store import InMemoryMemoryStore


def setup_standard_store() -> InMemoryMemoryStore:
    """Populates standard in-memory test store with realistic persistent memories."""
    store = InMemoryMemoryStore(embedding_model=DeterministicMockEmbedding(dimension=64))

    memories = [
        Memory(
            id="mem-1",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Likes gaming",
            topic="gaming",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem-2",
            persona_id="vivek",
            memory_type=MemoryType.EXPERIENCE,
            content="Participated in a gaming tournament",
            topic="gaming",
            importance=ImportanceLevel.MEDIUM,
        ),
        Memory(
            id="mem-3",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Likes Marvel movies",
            topic="movies",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem-4",
            persona_id="vivek",
            memory_type=MemoryType.FACT,
            content="Studies B.Tech CSE",
            topic="college",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem-5",
            persona_id="vivek",
            memory_type=MemoryType.FACT,
            content="Owns an Acer Nitro laptop",
            topic="technology",
            importance=ImportanceLevel.MEDIUM,
        ),
        Memory(
            id="mem-6",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Prefers Java",
            topic="technology",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem-7",
            persona_id="vivek",
            memory_type=MemoryType.RELATIONSHIP,
            content="Rahul is a college friend",
            topic="college",
            importance=ImportanceLevel.HIGH,
        ),
    ]

    for m in memories:
        store.save(m)

    return store


def run_benchmark_1(retriever: MemoryRetriever) -> bool:
    """Benchmark 1: Gaming retrieval."""
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Gaming Retrieval")
    print("Conversation: 'bhai bgmi khelega?'")

    ranked = retriever.retrieve(
        persona_id="vivek",
        query="bhai bgmi khelega?",
        top_k=3,
    )

    for idx, r in enumerate(ranked, 1):
        print(f"  [{idx}] Score={r.score:.4f} | Type={r.memory.memory_type.value:12} | Topic={r.memory.topic:10} | Content={r.memory.content}")

    # Top memories must be gaming-related
    top_topics = [r.memory.topic for r in ranked[:2]]
    success = top_topics[0] == "gaming" and "gaming" in top_topics
    print(f"Result: {'PASS' if success else 'FAIL'} (Top topic is {ranked[0].memory.topic})")
    return success


def run_benchmark_2(retriever: MemoryRetriever) -> bool:
    """Benchmark 2: College retrieval."""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: College Retrieval")
    print("Conversation: 'Dean ne attendance ka notice nikala hai'")

    ranked = retriever.retrieve(
        persona_id="vivek",
        query="Dean ne attendance ka notice nikala hai",
        top_k=3,
    )

    for idx, r in enumerate(ranked, 1):
        print(f"  [{idx}] Score={r.score:.4f} | Type={r.memory.memory_type.value:12} | Topic={r.memory.topic:10} | Content={r.memory.content}")

    success = ranked[0].memory.topic == "college"
    print(f"Result: {'PASS' if success else 'FAIL'} (Top topic is {ranked[0].memory.topic})")
    return success


def run_benchmark_3(retriever: MemoryRetriever) -> bool:
    """Benchmark 3: Technology retrieval."""
    print("\n" + "=" * 60)
    print("BENCHMARK 3: Technology Retrieval")
    print("Conversation: 'bhai battery bahut jaldi drain ho rahi'")

    ranked = retriever.retrieve(
        persona_id="vivek",
        query="bhai battery bahut jaldi drain ho rahi",
        top_k=3,
    )

    for idx, r in enumerate(ranked, 1):
        print(f"  [{idx}] Score={r.score:.4f} | Type={r.memory.memory_type.value:12} | Topic={r.memory.topic:10} | Content={r.memory.content}")

    success = ranked[0].memory.topic == "technology"
    print(f"Result: {'PASS' if success else 'FAIL'} (Top topic is {ranked[0].memory.topic})")
    return success


def run_benchmark_4(retriever: MemoryRetriever) -> bool:
    """Benchmark 4: Movie preferences retrieval."""
    print("\n" + "=" * 60)
    print("BENCHMARK 4: Movie Preferences Retrieval")
    print("Conversation: 'Marvel ki new movie dekhi?'")

    ranked = retriever.retrieve(
        persona_id="vivek",
        query="Marvel ki new movie dekhi?",
        top_k=3,
    )

    for idx, r in enumerate(ranked, 1):
        print(f"  [{idx}] Score={r.score:.4f} | Type={r.memory.memory_type.value:12} | Topic={r.memory.topic:10} | Content={r.memory.content}")

    success = ranked[0].memory.topic == "movies" and "Marvel" in ranked[0].memory.content
    print(f"Result: {'PASS' if success else 'FAIL'} (Top memory is '{ranked[0].memory.content}')")
    return success


def run_benchmark_5(retriever: MemoryRetriever) -> bool:
    """
    Benchmark 5: Contextual disambiguation for ambiguous turn ('Aaja').
    Context A: Gaming session context -> Gaming memories prioritized.
    Context B: Canteen / College context -> Gaming memories DO NOT dominate.
    """
    print("\n" + "=" * 60)
    print("BENCHMARK 5: Ambiguous Turn Disambiguation ('Aaja')")

    # Context A: Gaming
    ctx_gaming = ContextAnalysis(
        topic=TopicCategory.GAMING,
        conversation_state=ConversationState.ONGOING,
        user_intent=UserIntent.INVITATION,
        ambiguity=AmbiguityLevel.HIGH,
        recent_activity="Invited to play online match",
        context_depth=3,
        last_user_message="Aaja",
        speaker_alternation_rate=1.0,
    )

    ranked_gaming = retriever.retrieve(
        persona_id="vivek",
        context_analysis=ctx_gaming,
        top_k=3,
    )

    print("\n--- Context A: Topic=Gaming ---")
    for idx, r in enumerate(ranked_gaming, 1):
        print(f"  [{idx}] Score={r.score:.4f} | Topic={r.memory.topic:10} | Content={r.memory.content}")

    # Context B: College / Canteen
    ctx_college = ContextAnalysis(
        topic=TopicCategory.COLLEGE,
        conversation_state=ConversationState.ONGOING,
        user_intent=UserIntent.INVITATION,
        ambiguity=AmbiguityLevel.HIGH,
        recent_activity="Discussing college break / canteen",
        context_depth=3,
        last_user_message="Aaja",
        speaker_alternation_rate=1.0,
    )

    ranked_college = retriever.retrieve(
        persona_id="vivek",
        context_analysis=ctx_college,
        top_k=3,
    )

    print("\n--- Context B: Topic=College ---")
    for idx, r in enumerate(ranked_college, 1):
        print(f"  [{idx}] Score={r.score:.4f} | Topic={r.memory.topic:10} | Content={r.memory.content}")

    # Assertions:
    # Under gaming context, top memory is gaming
    pass_a = ranked_gaming[0].memory.topic == "gaming"
    # Under college context, top memory is college (gaming does not dominate)
    pass_b = ranked_college[0].memory.topic == "college" and ranked_college[0].score > ranked_college[-1].score

    success = pass_a and pass_b
    print(f"\nResult: {'PASS' if success else 'FAIL'} (Gaming context top={ranked_gaming[0].memory.topic}, College context top={ranked_college[0].memory.topic})")
    return success


def main():
    print("=" * 60)
    print("STAGE 6C -- MEMORY ENGINE INTEGRATION BENCHMARKS")
    print("=" * 60)

    store = setup_standard_store()
    retriever = MemoryRetriever(store=store)

    results = [
        run_benchmark_1(retriever),
        run_benchmark_2(retriever),
        run_benchmark_3(retriever),
        run_benchmark_4(retriever),
        run_benchmark_5(retriever),
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"BENCHMARK SUMMARY: {passed}/{total} Passed")
    print("=" * 60)

    if passed == total:
        print("ALL 5 BENCHMARKS PASSED SUCCESSFULLY.")
        sys.exit(0)
    else:
        print("SOME BENCHMARKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
