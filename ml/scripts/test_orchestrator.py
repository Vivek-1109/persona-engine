"""
Integration Benchmark for Persona Engine Conversation Orchestrator (Stage 6D).
Tests 10 realistic conversation flows:
1. Gaming invitation ('bhai bgmi khelega?')
2. College discussion ('Dean ne attendance ka notice nikala hai')
3. Technology problem ('bhai battery bahut jaldi drain ho rahi')
4. Movie conversation ('Marvel ki new movie dekhi kya?')
5. Ambiguous 'Aaja' (isolated turn without context)
6. Contextual 'Aaja' (with preceding gaming dialogue context)
7. 'Khelega?' (direct invitation)
8. 'Thik' (quick agreement)
9. 'Nhi bhai' (quick disagreement/negative reaction)
10. Closing conversation ('Chal baad me baat karta hu')
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.context.context_analyzer import ContextAnalyzer
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
from ml.src.orchestrator.orchestrator import ConversationOrchestrator
from ml.src.orchestrator.orchestrator_schema import (
    ConversationRequest,
    ResponsePlan,
    ResponseStrategy,
    ResponseTone,
)


def setup_benchmark_environment() -> ConversationOrchestrator:
    """Prepares an Orchestrator instance wired to ContextAnalyzer and populated MemoryRetriever."""
    embedding_model = DeterministicMockEmbedding(dimension=64)
    store = InMemoryMemoryStore(embedding_model=embedding_model)

    test_memories = [
        Memory(
            id="mem_gaming_pref",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Likes gaming",
            topic="gaming",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem_gaming_exp",
            persona_id="vivek",
            memory_type=MemoryType.EXPERIENCE,
            content="Participated in a gaming tournament",
            topic="gaming",
            importance=ImportanceLevel.MEDIUM,
        ),
        Memory(
            id="mem_college_fact",
            persona_id="vivek",
            memory_type=MemoryType.FACT,
            content="Studies B.Tech CSE",
            topic="college",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem_movie_pref",
            persona_id="vivek",
            memory_type=MemoryType.PREFERENCE,
            content="Likes Marvel movies",
            topic="movies",
            importance=ImportanceLevel.HIGH,
        ),
        Memory(
            id="mem_tech_fact",
            persona_id="vivek",
            memory_type=MemoryType.FACT,
            content="Owns an Acer Nitro laptop",
            topic="technology",
            importance=ImportanceLevel.MEDIUM,
        ),
    ]

    for m in test_memories:
        store.save(m)

    retriever = MemoryRetriever(
        store=store,
        ranker=MemoryRanker(),
        embedding_model=embedding_model,
    )

    return ConversationOrchestrator(
        context_analyzer=ContextAnalyzer(),
        memory_retriever=retriever,
    )


def print_plan_summary(index: int, title: str, utterance: str, plan: ResponsePlan) -> None:
    """Prints structured test case results without exposing sensitive data."""
    print(f"\n[{index}] {title}")
    print(f"    Utterance: '{utterance}'")
    print(f"    Topic:       {plan.topic.value}")
    print(f"    Intent:      {plan.user_intent.value}")
    print(f"    State:       {plan.conversation_state.value}")
    print(f"    Ambiguity:   {plan.ambiguity.value}")
    print(f"    Strategy:    {plan.response_strategy.value}")
    print(f"    Tone:        {plan.tone.value}")
    print(f"    Use Memory:  {plan.use_memory} (IDs: {plan.selected_memory_ids})")
    print(f"    Confidence:  {plan.confidence:.2f}")


def run_benchmarks() -> bool:
    orchestrator = setup_benchmark_environment()
    results: List[Tuple[str, bool]] = []

    print("=" * 65)
    print("STAGE 6D -- CONVERSATION ORCHESTRATOR INTEGRATION BENCHMARKS")
    print("=" * 65)

    # 1. Gaming invitation
    req1 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_1",
        messages=[{"role": "user", "content": "bhai bgmi khelega?"}],
    )
    plan1 = orchestrator.plan(req1)
    print_plan_summary(1, "Gaming Invitation", "bhai bgmi khelega?", plan1)
    pass1 = (
        plan1.topic == TopicCategory.GAMING
        and plan1.response_strategy == ResponseStrategy.ACCEPT
        and plan1.tone == ResponseTone.CASUAL
    )
    results.append(("1. Gaming Invitation", pass1))

    # 2. College discussion
    req2 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_2",
        messages=[{"role": "user", "content": "Dean ne attendance ka notice nikala hai"}],
    )
    plan2 = orchestrator.plan(req2)
    print_plan_summary(2, "College Discussion", "Dean ne attendance ka notice nikala hai", plan2)
    pass2 = (
        plan2.topic == TopicCategory.COLLEGE
        and plan2.response_strategy in {ResponseStrategy.PROVIDE_INFORMATION, ResponseStrategy.ANSWER}
        and plan2.tone == ResponseTone.SERIOUS
    )
    results.append(("2. College Discussion", pass2))

    # 3. Technology problem
    req3 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_3",
        messages=[{"role": "user", "content": "bhai battery bahut jaldi drain ho rahi"}],
    )
    plan3 = orchestrator.plan(req3)
    print_plan_summary(3, "Technology Problem", "bhai battery bahut jaldi drain ho rahi", plan3)
    pass3 = (
        plan3.topic == TopicCategory.TECHNOLOGY
        and plan3.response_strategy in {ResponseStrategy.SUGGEST, ResponseStrategy.ANSWER}
        and plan3.tone == ResponseTone.SUPPORTIVE
    )
    results.append(("3. Technology Problem", pass3))

    # 4. Movie conversation
    req4 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_4",
        messages=[{"role": "user", "content": "Marvel ki new movie dekhi kya?"}],
    )
    plan4 = orchestrator.plan(req4)
    print_plan_summary(4, "Movie Conversation", "Marvel ki new movie dekhi kya?", plan4)
    pass4 = (
        plan4.topic == TopicCategory.MOVIES
        and plan4.response_strategy in {ResponseStrategy.ANSWER, ResponseStrategy.REACT, ResponseStrategy.PROVIDE_INFORMATION}
        and plan4.tone == ResponseTone.CASUAL
    )
    results.append(("4. Movie Conversation", pass4))

    # 5. Ambiguous "Aaja" (Isolated turn)
    req5 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_5",
        messages=[{"role": "user", "content": "Aaja"}],
    )
    plan5 = orchestrator.plan(req5)
    print_plan_summary(5, "Ambiguous 'Aaja'", "Aaja", plan5)
    pass5 = (
        plan5.ambiguity == AmbiguityLevel.HIGH
        and plan5.response_strategy in {ResponseStrategy.ASK_CLARIFICATION, ResponseStrategy.REACT}
    )
    results.append(("5. Ambiguous 'Aaja'", pass5))

    # 6. Contextual "Aaja" (Preceding gaming dialogue)
    req6 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_6",
        messages=[
            {"role": "user", "content": "bhai bgmi update aa gaya"},
            {"role": "assistant", "content": "haan dekha maine"},
            {"role": "user", "content": "Aaja"},
        ],
    )
    plan6 = orchestrator.plan(req6)
    print_plan_summary(6, "Contextual 'Aaja'", "Aaja (with gaming context)", plan6)
    pass6 = (
        plan6.topic == TopicCategory.GAMING
        and plan6.response_strategy == ResponseStrategy.ACCEPT
    )
    results.append(("6. Contextual 'Aaja'", pass6))

    # 7. "Khelega?"
    req7 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_7",
        messages=[{"role": "user", "content": "Khelega?"}],
    )
    plan7 = orchestrator.plan(req7)
    print_plan_summary(7, "Direct Invitation 'Khelega?'", "Khelega?", plan7)
    pass7 = (
        plan7.topic == TopicCategory.GAMING
        and plan7.response_strategy == ResponseStrategy.ACCEPT
    )
    results.append(("7. 'Khelega?'", pass7))

    # 8. "Thik"
    req8 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_8",
        messages=[
            {"role": "assistant", "content": "kal milte hain college me"},
            {"role": "user", "content": "thik"},
        ],
    )
    plan8 = orchestrator.plan(req8)
    print_plan_summary(8, "Quick Agreement 'Thik'", "thik", plan8)
    pass8 = plan8.response_strategy == ResponseStrategy.ACKNOWLEDGE
    results.append(("8. 'Thik'", pass8))

    # 9. "Nhi bhai"
    req9 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_9",
        messages=[
            {"role": "assistant", "content": "chal ghumne chalte hain"},
            {"role": "user", "content": "nhi bhai"},
        ],
    )
    plan9 = orchestrator.plan(req9)
    print_plan_summary(9, "Decline/Reaction 'Nhi bhai'", "nhi bhai", plan9)
    pass9 = plan9.response_strategy in {ResponseStrategy.REACT, ResponseStrategy.DECLINE}
    results.append(("9. 'Nhi bhai'", pass9))

    # 10. Closing conversation
    req10 = ConversationRequest(
        persona_id="vivek",
        conversation_id="bench_10",
        messages=[
            {"role": "assistant", "content": "ok bhai"},
            {"role": "user", "content": "Chal baad me baat karta hu"},
        ],
    )
    plan10 = orchestrator.plan(req10)
    print_plan_summary(10, "Closing Conversation", "Chal baad me baat karta hu", plan10)
    pass10 = plan10.response_strategy == ResponseStrategy.CLOSE_CONVERSATION
    results.append(("10. Closing Conversation", pass10))

    # Summary
    print("\n" + "=" * 65)
    print("BENCHMARK EXECUTION SUMMARY:")
    print("=" * 65)
    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name:35} : {status}")
        if not success:
            all_passed = False

    passed_count = sum(1 for _, s in results if s)
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} benchmarks passed.")
    print("=" * 65)

    return all_passed


if __name__ == "__main__":
    success = run_benchmarks()
    sys.exit(0 if success else 1)
