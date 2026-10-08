"""
Strongly-typed schemas and data models for Stage 6E End-to-End Evaluation & Benchmarking.
Defines BenchmarkCase, BenchmarkResult, EvaluationSummary, and AblationMode.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkCategory(str, Enum):
    """Controlled taxonomy of evaluation benchmark categories."""
    CASUAL_CONVERSATION = "casual_conversation"
    GAMING = "gaming"
    COLLEGE = "college"
    TECHNOLOGY = "technology"
    MOVIES = "movies"
    PLANNING = "planning"
    SOCIAL = "social"
    AMBIGUOUS_SHORT_MESSAGES = "ambiguous_short_messages"
    MULTI_TURN_CONTEXT = "multi_turn_context"
    MEMORY_DEPENDENT = "memory_dependent"
    MEMORY_IRRELEVANT = "memory_irrelevant"
    CLOSING = "closing"
    DISAGREEMENT_REJECTION = "disagreement_rejection"
    QUESTIONS = "questions"
    INVITATIONS = "invitations"
    UNSEEN_GENERALIZATION = "unseen_generalization"


class ErrorType(str, Enum):
    """Standardized error classifications for failure analysis."""
    CONTEXT_ERROR = "CONTEXT_ERROR"
    INTENT_ERROR = "INTENT_ERROR"
    MEMORY_RETRIEVAL_ERROR = "MEMORY_RETRIEVAL_ERROR"
    MEMORY_CONTAMINATION = "MEMORY_CONTAMINATION"
    STRATEGY_ERROR = "STRATEGY_ERROR"
    PERSONA_STYLE_ERROR = "PERSONA_STYLE_ERROR"
    GENERATION_ERROR = "GENERATION_ERROR"
    HALLUCINATION = "HALLUCINATION"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    OTHER = "OTHER"


class AblationMode(str, Enum):
    """Component ablation modes for isolating subsystem contributions."""
    FULL_SYSTEM = "full_system"
    NO_MEMORY = "no_memory"
    NO_CONTEXT = "no_context"
    NO_ORCHESTRATOR = "no_orchestrator"
    PERSONA_ONLY = "persona_only"


class BenchmarkCase(BaseModel):
    """A synthetic or safely-derived dialogue test case for end-to-end evaluation."""
    case_id: str = Field(..., description="Unique case identifier (e.g. 'CASE_GAMING_01').")
    category: BenchmarkCategory = Field(..., description="Target evaluation category.")
    conversation: List[Dict[str, str]] = Field(
        ...,
        description="Chronological dialogue history [{'role': 'user'|'assistant', 'content': '...'}]."
    )
    expected_topic: str = Field(..., description="Ground-truth topic category.")
    expected_intent: str = Field(..., description="Ground-truth user intent.")
    expected_strategy: str = Field(..., description="Ground-truth response strategy.")
    expected_context_behavior: Optional[str] = Field(
        default=None,
        description="Expected conversational context handling notes."
    )
    expected_memory_behavior: Optional[str] = Field(
        default=None,
        description="Expected memory behavior description."
    )
    expected_use_memory: Optional[bool] = Field(
        default=None,
        description="Whether memory is expected to be utilized in this turn."
    )
    available_memories: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Mock/existing memories seeded in the store for this test case."
    )
    evaluation_notes: Optional[str] = Field(
        default=None,
        description="Qualitative commentary and acceptance criteria."
    )


class BenchmarkResult(BaseModel):
    """Structured evaluation output for a single benchmark execution."""
    case_id: str
    category: BenchmarkCategory
    turn_count: int
    last_user_message: str

    # Ground truth vs predictions
    expected_topic: str
    predicted_topic: str
    topic_correct: bool

    expected_intent: str
    predicted_intent: str
    intent_correct: bool

    expected_strategy: str
    predicted_strategy: str
    strategy_correct: bool

    # Ambiguity evaluation
    ambiguity_level: str
    ambiguity_handled_correctly: bool

    # Memory evaluation
    use_memory: bool
    expected_use_memory: Optional[bool]
    memory_behavior_correct: bool
    retrieved_memory_ids: List[str]
    selected_memory_ids: List[str]
    memory_contamination: bool

    # Strategy & planning outputs
    response_tone: str
    confidence: float
    generation_instruction: str

    # Final response analysis
    final_response: str
    response_chars: int
    response_words: int
    is_hinglish: bool
    has_emoji: bool
    has_slang: bool
    has_hallucination: bool

    # Error classification
    error_types: List[ErrorType] = Field(default_factory=list)
    failure_explanation: Optional[str] = None

    # Component latencies in milliseconds
    latencies_ms: Dict[str, float] = Field(
        default_factory=lambda: {
            "context_ms": 0.0,
            "memory_ms": 0.0,
            "orchestrator_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return self.model_dump()


class EvaluationSummary(BaseModel):
    """Aggregated quantitative metrics across the entire benchmark suite."""
    total_cases: int
    category_counts: Dict[str, int]

    # Core accuracies
    topic_accuracy: float
    intent_accuracy: float
    strategy_accuracy: float
    ambiguity_accuracy: float

    # Memory metrics
    memory_precision: float
    memory_recall: float
    memory_contamination_rate: float
    memory_utilization_rate: float

    # Persona & style metrics
    hinglish_rate: float
    slang_rate: float
    emoji_rate: float
    hallucination_rate: float

    # Response length distribution
    length_distribution: Dict[str, float]  # mean, median, p25, p75, p90

    # Latency distribution
    latency_summary: Dict[str, Dict[str, float]]

    # Category performance breakdown
    category_metrics: Dict[str, Dict[str, float]]

    # Error type frequency
    error_breakdown: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return self.model_dump()
