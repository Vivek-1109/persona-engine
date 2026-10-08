"""
Memory policy engine for Persona Engine Conversation Orchestrator (Stage 6D).
Filters, vets, and selects candidate memories for generation payload inclusion.
Prevents prompt pollution and ensures persona naturalness without forced memory citation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ml.src.context.context_schema import TopicCategory
from ml.src.memory.memory_ranker import RankedMemory
from ml.src.memory.memory_schema import Memory, MemoryStatus


class MemoryPolicy:
    """
    Enforces quality, relevance, and safety gates on retrieved memories.
    Consumes Stage 6C ranking without recreating it.
    """

    DEFAULT_RELEVANCE_THRESHOLD = 0.45
    MAX_SELECTED_MEMORIES = 2

    @classmethod
    def evaluate_memories(
        cls,
        candidate_memories: Optional[List[Union[RankedMemory, Memory]]],
        current_topic: TopicCategory,
        conversation_state: Optional[Any] = None,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        max_items: int = MAX_SELECTED_MEMORIES,
    ) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
        """
        Evaluates candidate memories against the current conversational topic and relevance scores.

        Args:
            candidate_memories: RankedMemory or Memory objects from Stage 6C.
            current_topic: Topic category active in the current conversation.
            conversation_state: Current conversation state (closing states suppress memories).
            threshold: Minimum composite ranking score required for inclusion.
            max_items: Maximum number of memories allowed to pass to generation.

        Returns:
            Tuple of (use_memory: bool, selected_memory_ids: List[str], selected_memories_summary: List[Dict])
        """
        if not candidate_memories:
            return False, [], []

        # Closing state should not dredge up memories
        state_str = str(getattr(conversation_state, "value", conversation_state) or "").lower()
        if state_str == "closing":
            return False, [], []

        selected_ids: List[str] = []
        selected_summaries: List[Dict[str, Any]] = []

        topic_str = current_topic.value.lower() if hasattr(current_topic, "value") else str(current_topic).lower()

        for item in candidate_memories:
            # Extract memory and score
            if isinstance(item, RankedMemory):
                mem = item.memory
                score = item.score
            elif isinstance(item, Memory):
                mem = item
                mem_topic = (mem.topic or "").lower()
                score = 0.60 if (mem_topic and mem_topic == topic_str) else 0.30
            else:
                continue

            # 1. Active status check
            if mem.status != MemoryStatus.ACTIVE:
                continue

            # 2. Score threshold check
            if score < threshold:
                continue

            # 3. Topic relevance filter: memory topic must match current conversation domain
            mem_topic = (mem.topic or "").lower()
            if mem_topic:
                if topic_str in {"casual_chat", "other"}:
                    if mem_topic not in {"casual_chat", "general"}:
                        continue
                elif mem_topic != topic_str:
                    continue

            # 4. Acceptance
            selected_ids.append(mem.id)
            selected_summaries.append({
                "id": mem.id,
                "memory_type": mem.memory_type.value,
                "content": mem.content,
                "topic": mem.topic,
                "importance": mem.importance.value,
                "score": round(score, 4),
            })

            if len(selected_ids) >= max_items:
                break

        use_memory = len(selected_ids) > 0
        return use_memory, selected_ids, selected_summaries
