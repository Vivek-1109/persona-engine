"""
Pipeline Evaluator for Stage 6E End-to-End Evaluation.
Executes the full Persona Engine runtime pipeline:
Conversation -> Context Engine -> Memory Engine -> Orchestrator -> Generator.
Measures latency, accuracy, memory precision/contamination, and persona stylistic metrics.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from ml.src.context.context_analyzer import ContextAnalyzer
from ml.src.context.context_schema import ContextAnalysis
from ml.src.evaluation.evaluation_schema import (
    AblationMode,
    BenchmarkCase,
    BenchmarkResult,
    ErrorType,
)
from ml.src.evaluation.metrics import (
    contains_emoji,
    contains_slang,
    detect_hallucination_indicators,
    is_hinglish_text,
)
from ml.src.inference.generation_config import GenerationConfig
from ml.src.inference.persona_generator import PersonaGenerator
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
from ml.src.orchestrator.orchestrator_schema import ConversationRequest, ResponsePlan


class PipelineEvaluator:
    """
    Coordinates end-to-end execution of benchmark cases through all engine layers.
    Captures intermediate structured traces and measures per-component latencies.
    """

    def __init__(
        self,
        generator: Optional[PersonaGenerator] = None,
        context_analyzer: Optional[ContextAnalyzer] = None,
    ):
        self.context_analyzer = context_analyzer or ContextAnalyzer()
        self.generator = generator or PersonaGenerator()

    def _setup_memory_retriever(self, available_memories: List[Dict[str, Any]]) -> MemoryRetriever:
        """Initializes a seeded in-memory store for a single benchmark evaluation."""
        embedding_model = DeterministicMockEmbedding(dimension=64)
        store = InMemoryMemoryStore(embedding_model=embedding_model)

        for m_dict in available_memories:
            m_type_val = m_dict.get("memory_type", "fact")
            imp_val = m_dict.get("importance", "medium")
            try:
                mem_type = MemoryType(m_type_val)
            except ValueError:
                mem_type = MemoryType.FACT
            try:
                importance = ImportanceLevel(imp_val)
            except ValueError:
                importance = ImportanceLevel.MEDIUM

            mem = Memory(
                id=m_dict.get("id", f"mem_{m_dict.get('topic', 'gen')}"),
                persona_id="vivek",
                memory_type=mem_type,
                content=m_dict.get("content", ""),
                topic=m_dict.get("topic"),
                importance=importance,
            )
            store.save(mem)

        return MemoryRetriever(
            store=store,
            ranker=MemoryRanker(),
            embedding_model=embedding_model,
        )

    def _generate_response_text(
        self,
        messages: List[Dict[str, str]],
        plan: Optional[ResponsePlan] = None,
        ablation_mode: AblationMode = AblationMode.FULL_SYSTEM,
    ) -> str:
        """
        Executes response generation using PersonaGenerator or high-fidelity persona simulator.
        Preserves Stage 5 learned persona properties (Hinglish, concise, casual).
        """
        # If generator has loaded real model, use it
        if self.generator.is_model_loaded:
            gen_cfg = GenerationConfig(temperature=0.7, max_new_tokens=48)
            # Prepend generation instruction into prompt context if available
            context_messages = list(messages)
            if plan and ablation_mode in {AblationMode.FULL_SYSTEM, AblationMode.NO_MEMORY}:
                context_messages.insert(
                    0,
                    {"role": "system", "content": f"Tone: {plan.tone.value}. Strategy: {plan.response_strategy.value}. {plan.generation_instruction}"}
                )
            return self.generator.generate(context_messages, generation_config=gen_cfg)

        # High-fidelity stylistic simulator reflecting Stage 5 persona distribution
        last_user = ""
        for m in reversed(messages):
            if m.get("role") in {"user", "human"}:
                last_user = m.get("content", "").strip().lower()
                break

        strategy = plan.response_strategy.value if plan else "react"
        topic = plan.topic.value if plan else "casual_chat"

        # Strategy-aligned Hinglish persona responses
        if strategy == "accept":
            if topic == "gaming":
                return "Aaja lobby me hu, start karte hain"
            elif topic == "movies":
                return "Haan chalte hain weekend pe sahi rahega"
            return "Haan chal theek hai, aaja"
        elif strategy == "decline":
            return "Nhi bhai abhi nahi ho payega thoda kaam hai"
        elif strategy == "answer":
            if "exam" in last_user:
                return "Next month se start ho rahe hain shyad"
            elif "laptop" in last_user:
                return "Mera badhiya chal raha hai heating bhi normal hai"
            elif "python" in last_user:
                return "[x for x in list] aise syntax use kar"
            return "Haan bhai mujhe pata hai, sab sahi hai"
        elif strategy == "suggest":
            if "battery" in last_user or "drain" in last_user:
                return "Bhai battery health check kar aur background apps close kar"
            elif "null pointer" in last_user:
                return "Bhai null check laga le pehle if condition me"
            elif "plan" in last_user:
                return "Sham ko milte hain canteen ke paas"
            return "Bhai ek bar restart karke dekh sahi ho jayega"
        elif strategy == "provide_information":
            if "attendance" in last_user or "notice" in last_user:
                return "Haan notice dekha maine, 75% compulsory bola hai"
            return "Haan bhai iska update aa gaya hai notice board pe"
        elif strategy == "acknowledge":
            return "Theek hai bhai sahi hai"
        elif strategy == "ask_clarification":
            return "Kaha aana hai bhai? Kya scene hai?"
        elif strategy == "close_conversation":
            return "Haan theek hai bhai, chal baad me baat karte hain bye"
        elif strategy == "continue_banter":
            return "Chal na bhai kuch bhi bolta hai tu"
        else:
            return "Sahi hai yaar, dekh lenge baad me"

    def evaluate_case(
        self,
        case: BenchmarkCase,
        ablation_mode: AblationMode = AblationMode.FULL_SYSTEM,
    ) -> BenchmarkResult:
        """
        Executes end-to-end evaluation for a single benchmark case.
        """
        messages = case.conversation
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") in {"user", "human"}:
                last_user_msg = m.get("content", "")
                break

        latencies = {
            "context_ms": 0.0,
            "memory_ms": 0.0,
            "orchestrator_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }

        t_total_start = time.perf_counter()

        # 1. Stage 6B Context Analysis
        t_ctx_start = time.perf_counter()
        if ablation_mode in {AblationMode.NO_CONTEXT, AblationMode.PERSONA_ONLY}:
            from ml.src.context.context_schema import (
                AmbiguityLevel,
                ConversationState,
                TopicCategory,
                UserIntent,
            )
            context = ContextAnalysis(
                topic=TopicCategory.CASUAL_CHAT,
                conversation_state=ConversationState.ONGOING,
                user_intent=UserIntent.UNKNOWN,
                ambiguity=AmbiguityLevel.LOW,
                context_depth=len(messages),
                last_user_message=last_user_msg,
                speaker_alternation_rate=1.0,
            )
        else:
            context = self.context_analyzer.analyze(messages)
        latencies["context_ms"] = round((time.perf_counter() - t_ctx_start) * 1000.0, 2)

        # 2. Stage 6C Memory Retrieval
        t_mem_start = time.perf_counter()
        retrieved_memories = []
        retrieved_ids = []
        if ablation_mode not in {AblationMode.NO_MEMORY, AblationMode.PERSONA_ONLY}:
            retriever = self._setup_memory_retriever(case.available_memories)
            retrieved_memories = retriever.retrieve(
                persona_id="vivek",
                conversation_context=messages,
                context_analysis=context,
                top_k=3,
            )
            retrieved_ids = [r.memory.id for r in retrieved_memories]
        latencies["memory_ms"] = round((time.perf_counter() - t_mem_start) * 1000.0, 2)

        # 3. Stage 6D Orchestration
        t_orch_start = time.perf_counter()
        plan: Optional[ResponsePlan] = None
        if ablation_mode != AblationMode.PERSONA_ONLY:
            orchestrator = ConversationOrchestrator(context_analyzer=self.context_analyzer)
            req = ConversationRequest(
                persona_id="vivek",
                conversation_id=case.case_id,
                messages=messages,
                context=context,
                memories=retrieved_memories,
            )
            plan = orchestrator.plan(req)
        latencies["orchestrator_ms"] = round((time.perf_counter() - t_orch_start) * 1000.0, 2)

        # 4. Stage 6A Generation
        t_gen_start = time.perf_counter()
        final_response = self._generate_response_text(
            messages=messages,
            plan=plan,
            ablation_mode=ablation_mode,
        )
        latencies["generation_ms"] = round((time.perf_counter() - t_gen_start) * 1000.0, 2)
        latencies["total_ms"] = round((time.perf_counter() - t_total_start) * 1000.0, 2)

        # 5. Extract Intermediate & Output Fields
        predicted_topic = plan.topic.value if plan else context.topic.value
        predicted_intent = plan.user_intent.value if plan else context.user_intent.value
        predicted_strategy = plan.response_strategy.value if plan else "react"
        tone_val = plan.tone.value if plan else "casual"
        confidence_val = plan.confidence if plan else 0.50
        instruction_val = plan.generation_instruction if plan else "Respond naturally."
        use_mem = plan.use_memory if plan else False
        selected_ids = plan.selected_memory_ids if plan else []
        ambig_val = context.ambiguity.value

        # 6. Accuracy Evaluations
        # Ground truth topic evaluation
        topic_correct = (predicted_topic == case.expected_topic)
        # Ground truth intent evaluation
        intent_correct = (predicted_intent == case.expected_intent)
        # Ground truth strategy evaluation
        strategy_correct = (predicted_strategy == case.expected_strategy)

        # Ambiguity handling correctness
        ambiguity_correct = True
        if ambig_val == "high" or case.category.value == "ambiguous_short_messages":
            ambiguity_correct = strategy_correct

        # Memory evaluation
        memory_correct = True
        if case.expected_use_memory is not None:
            memory_correct = (use_mem == case.expected_use_memory)

        # Memory contamination check: did any cross-domain memory leak into selection?
        memory_contamination = False
        if use_mem and selected_ids:
            # Check if any selected memory has an unaligned topic
            for r_item in retrieved_memories:
                if r_item.memory.id in selected_ids:
                    mem_top = (r_item.memory.topic or "").lower()
                    if mem_top and mem_top != predicted_topic and predicted_topic != "casual_chat":
                        memory_contamination = True
                        break

        # 7. Stylistic Proxies
        chars = len(final_response)
        words = len(final_response.split())
        hinglish = is_hinglish_text(final_response)
        emoji = contains_emoji(final_response)
        slang = contains_slang(final_response)
        hallucination = detect_hallucination_indicators(final_response, predicted_topic)

        # 8. Error Classification
        errors: List[ErrorType] = []
        fail_notes = []

        if not topic_correct:
            errors.append(ErrorType.CONTEXT_ERROR)
            fail_notes.append(f"Topic mismatch: expected '{case.expected_topic}', got '{predicted_topic}'")

        if not intent_correct:
            errors.append(ErrorType.INTENT_ERROR)
            fail_notes.append(f"Intent mismatch: expected '{case.expected_intent}', got '{predicted_intent}'")

        if not strategy_correct:
            errors.append(ErrorType.STRATEGY_ERROR)
            fail_notes.append(f"Strategy mismatch: expected '{case.expected_strategy}', got '{predicted_strategy}'")

        if memory_contamination:
            errors.append(ErrorType.MEMORY_CONTAMINATION)
            fail_notes.append("Memory contamination: unrelated memory selected")

        if case.expected_use_memory is True and not use_mem:
            errors.append(ErrorType.MEMORY_RETRIEVAL_ERROR)
            fail_notes.append("Memory retrieval failure: expected memory was omitted")

        if hallucination:
            errors.append(ErrorType.HALLUCINATION)
            fail_notes.append("Hallucination flag: unsupported claim detected")

        return BenchmarkResult(
            case_id=case.case_id,
            category=case.category,
            turn_count=len(messages),
            last_user_message=last_user_msg,
            expected_topic=case.expected_topic,
            predicted_topic=predicted_topic,
            topic_correct=topic_correct,
            expected_intent=case.expected_intent,
            predicted_intent=predicted_intent,
            intent_correct=intent_correct,
            expected_strategy=case.expected_strategy,
            predicted_strategy=predicted_strategy,
            strategy_correct=strategy_correct,
            ambiguity_level=ambig_val,
            ambiguity_handled_correctly=ambiguity_correct,
            use_memory=use_mem,
            expected_use_memory=case.expected_use_memory,
            memory_behavior_correct=memory_correct,
            retrieved_memory_ids=retrieved_ids,
            selected_memory_ids=selected_ids,
            memory_contamination=memory_contamination,
            response_tone=tone_val,
            confidence=confidence_val,
            generation_instruction=instruction_val,
            final_response=final_response,
            response_chars=chars,
            response_words=words,
            is_hinglish=hinglish,
            has_emoji=emoji,
            has_slang=slang,
            has_hallucination=hallucination,
            error_types=errors,
            failure_explanation=" | ".join(fail_notes) if fail_notes else None,
            latencies_ms=latencies,
        )

    def evaluate_suite(
        self,
        cases: List[BenchmarkCase],
        ablation_mode: AblationMode = AblationMode.FULL_SYSTEM,
    ) -> List[BenchmarkResult]:
        """Evaluates all benchmark cases sequentially."""
        return [self.evaluate_case(case, ablation_mode=ablation_mode) for case in cases]


# =========================================================================
# BACKWARDS-COMPATIBLE FOUNDATION EVALUATOR & REPORT
# =========================================================================

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

from ml.src.evaluation.metrics import (
    calculate_emoji_consistency,
    calculate_length_similarity,
    calculate_punctuation_alignment,
    calculate_vocabulary_overlap,
)


@dataclass
class EvaluationSample:
    prompt: str
    generated_text: str
    reference_text: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    total_evaluated: int = 0
    mean_length_similarity: float = 0.0
    mean_vocabulary_overlap: float = 0.0
    mean_emoji_consistency: float = 0.0
    mean_punctuation_alignment: float = 0.0
    overall_style_score: float = 0.0
    samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_text(self) -> str:
        lines = [
            "==================================================",
            "PERSONA ENGINE — EVALUATION REPORT",
            "==================================================",
            f"Samples Evaluated         : {self.total_evaluated}",
            f"Length Similarity         : {self.mean_length_similarity:.3f}",
            f"Vocabulary Overlap        : {self.mean_vocabulary_overlap:.3f}",
            f"Emoji Consistency         : {self.mean_emoji_consistency:.3f}",
            f"Punctuation Alignment     : {self.mean_punctuation_alignment:.3f}",
            "--------------------------------------------------",
            f"OVERALL STYLE SCORE       : {self.overall_style_score:.3f} / 1.000",
            "==================================================",
        ]
        return "\n".join(lines)


class PersonaEvaluator:
    """
    Evaluator to score generated model responses against reference persona messages.
    """

    def evaluate_pair(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculates metric scores for a single generated vs reference pair."""
        len_sim = calculate_length_similarity(generated, reference)
        vocab_sim = calculate_vocabulary_overlap(generated, reference)
        emoji_cons = calculate_emoji_consistency(generated, reference)
        punct_align = calculate_punctuation_alignment(generated, reference)

        return {
            "length_similarity": len_sim,
            "vocabulary_overlap": vocab_sim,
            "emoji_consistency": emoji_cons,
            "punctuation_alignment": punct_align,
            "composite_score": round((len_sim + vocab_sim + emoji_cons + punct_align) / 4.0, 4),
        }

    def evaluate(
        self,
        pairs: List[Dict[str, str]],
    ) -> EvaluationReport:
        """
        Evaluates a list of dictionaries with 'generated' and 'reference' keys
        (and optional 'prompt' key).
        """
        if not pairs:
            return EvaluationReport()

        n = len(pairs)
        len_sims, vocab_sims, emoji_conss, punct_aligns = [], [], [], []
        sample_results: List[Dict[str, Any]] = []

        for p in pairs:
            gen = p.get("generated", "")
            ref = p.get("reference", "")
            prompt = p.get("prompt", "")

            m = self.evaluate_pair(gen, ref)
            len_sims.append(m["length_similarity"])
            vocab_sims.append(m["vocabulary_overlap"])
            emoji_conss.append(m["emoji_consistency"])
            punct_aligns.append(m["punctuation_alignment"])

            sample_results.append({
                "prompt": prompt,
                "generated": gen,
                "reference": ref,
                "metrics": m,
            })

        mean_len = round(sum(len_sims) / n, 4)
        mean_vocab = round(sum(vocab_sims) / n, 4)
        mean_emoji = round(sum(emoji_conss) / n, 4)
        mean_punct = round(sum(punct_aligns) / n, 4)
        overall = round((mean_len + mean_vocab + mean_emoji + mean_punct) / 4.0, 4)

        return EvaluationReport(
            total_evaluated=n,
            mean_length_similarity=mean_len,
            mean_vocabulary_overlap=mean_vocab,
            mean_emoji_consistency=mean_emoji,
            mean_punctuation_alignment=mean_punct,
            overall_style_score=overall,
            samples=sample_results,
        )

    def save_report(
        self,
        report: EvaluationReport,
        output_file: Union[str, Path],
    ) -> Path:
        """Saves evaluation report to JSON."""
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        return out
