"""
Memory Retrieval Engine for Persona Engine (Stage 6C).
Retrieves and ranks relevant memories for a given persona and conversation context,
integrating with Stage 6B ContextAnalysis without modifying Context Engine.
"""

from typing import Any, Dict, List, Optional, Union

from ml.src.context.context_schema import ContextAnalysis
from ml.src.memory.embedding import BaseEmbeddingModel, DeterministicMockEmbedding
from ml.src.memory.memory_ranker import MemoryRanker, RankedMemory
from ml.src.memory.memory_schema import Memory, MemoryStatus
from ml.src.memory.memory_store import BaseMemoryStore, InMemoryMemoryStore


class MemoryRetriever:
    """
    Coordinates candidate retrieval, Stage 6B context integration,
    and deterministic ranking for persistent memories.
    """

    def __init__(
        self,
        store: Optional[BaseMemoryStore] = None,
        ranker: Optional[MemoryRanker] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
    ):
        self.embedding_model = embedding_model or DeterministicMockEmbedding()
        self.store = store or InMemoryMemoryStore(embedding_model=self.embedding_model)
        self.ranker = ranker or MemoryRanker()

    def retrieve(
        self,
        persona_id: str,
        query: Optional[str] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        context_analysis: Optional[Union[ContextAnalysis, Dict[str, Any]]] = None,
        topic: Optional[str] = None,
        top_k: int = 5,
        update_access_time: bool = True,
    ) -> List[RankedMemory]:
        """
        Retrieves top-K ranked memories relevant to the current conversation context.

        Args:
            persona_id: Target persona identifier for strict data isolation.
            query: Direct query string (optional if context_analysis or context provided).
            conversation_context: Multi-turn message history.
            context_analysis: Stage 6B ContextAnalysis instance or dict.
            topic: Explicit topic filter or boost.
            top_k: Maximum number of memories to return.
            update_access_time: Whether to update last_accessed_at in store.

        Returns:
            List of RankedMemory objects sorted by relevance score.
        """
        # Strict persona isolation
        active_memories = self.store.list_by_persona(persona_id=persona_id, status=MemoryStatus.ACTIVE)
        if not active_memories:
            return []

        # 1. Resolve query text
        resolved_query = (query or "").strip()
        context_topic = topic

        # 2. Consume Stage 6B ContextAnalysis if provided
        if context_analysis is not None:
            if isinstance(context_analysis, ContextAnalysis):
                analysis_dict = context_analysis.to_dict()
            else:
                analysis_dict = context_analysis

            # Extract topic if not explicitly overridden
            if not context_topic and "topic" in analysis_dict:
                topic_val = analysis_dict["topic"]
                context_topic = topic_val.value if hasattr(topic_val, "value") else str(topic_val)

            # Use last_user_message as query if no query provided
            if not resolved_query and "last_user_message" in analysis_dict:
                resolved_query = analysis_dict["last_user_message"]

        # 3. Fallback to conversation_context if query still empty
        if not resolved_query and conversation_context:
            for msg in reversed(conversation_context):
                if msg.get("role") in {"user", "human"}:
                    resolved_query = msg.get("content", "").strip()
                    break

        if not resolved_query:
            resolved_query = context_topic or ""

        # 4. Generate query embedding
        query_embedding = (
            self.embedding_model.embed_text(resolved_query)
            if resolved_query
            else None
        )

        # 5. Rank candidate memories
        ranked = self.ranker.rank_memories(
            memories=active_memories,
            query=resolved_query,
            query_embedding=query_embedding,
            context_topic=context_topic,
            top_k=top_k,
        )

        # 6. Update last accessed timestamps
        if update_access_time:
            for item in ranked:
                self.store.update_last_accessed(item.memory.id)

        return ranked

    def retrieve_memories(
        self,
        persona_id: str,
        query: Optional[str] = None,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        context_analysis: Optional[Union[ContextAnalysis, Dict[str, Any]]] = None,
        topic: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Memory]:
        """Convenience wrapper returning pure Memory objects without ranking breakdown."""
        ranked = self.retrieve(
            persona_id=persona_id,
            query=query,
            conversation_context=conversation_context,
            context_analysis=context_analysis,
            topic=topic,
            top_k=top_k,
        )
        return [item.memory for item in ranked]
