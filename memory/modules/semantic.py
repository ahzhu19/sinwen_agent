"""语义记忆模块。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..concept_extractor import ConceptExtractor, create_concept_extractor, extract_concepts
from ..config import MemoryConfig
from ..semantic_outbox_processor import SemanticOutboxProcessor
from .base import MemoryRecord


class SemanticMemory:
    """语义记忆：Neo4j 知识图谱 + Milvus 向量检索（Neo4j 内 Transactional Outbox）。"""

    memory_type = "semantic"

    def __init__(
        self,
        config: MemoryConfig,
        user_id: str,
        semantic_store: Any,
        vector_store: Any,
        embedding_provider: Any,
        concept_extractor: ConceptExtractor | None = None,
        semantic_outbox_processor: SemanticOutboxProcessor | None = None,
    ) -> None:
        self.config = config
        self.user_id = user_id
        self._store = semantic_store
        self._vectors = vector_store
        self._embeddings = embedding_provider
        self._concept_extractor = concept_extractor or create_concept_extractor(config)
        self._semantic_outbox_processor = semantic_outbox_processor
        self._ensure_neo4j_outbox()

    def _ensure_neo4j_outbox(self) -> None:
        if not hasattr(self._store, "write_memory_with_outbox"):
            raise ValueError("语义记忆 store 需实现 write_memory_with_outbox")

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        memory_id = str(uuid4())
        self._upsert_memory(
            memory_id,
            content=content,
            importance=importance,
            metadata=dict(metadata),
            operation="create",
        )
        return memory_id

    def update(
        self,
        memory_id: str,
        *,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        """原地更新语义记忆，保留 memory_id（Neo4j MERGE + Milvus 同 ID upsert）。"""
        if not self._store.get_many([memory_id]):
            raise KeyError(f"未找到记忆: {memory_id}")
        self._upsert_memory(
            memory_id,
            content=content,
            importance=importance,
            metadata=dict(metadata),
            operation="update",
        )
        return memory_id

    def _upsert_memory(
        self,
        memory_id: str,
        *,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        operation: str,
    ) -> None:
        payload = dict(metadata)
        extraction = extract_concepts(self._concept_extractor, content, payload)
        concepts = extraction.concepts
        payload["concept_extraction_source"] = extraction.source
        if extraction.error:
            payload["concept_extraction_error"] = extraction.error

        self._store.write_memory_with_outbox(
            user_id=self.user_id,
            memory_id=memory_id,
            content=content,
            importance=importance,
            metadata=payload,
            concepts=concepts,
            operation=operation,
            embedding_model=self.config.embed_model_name,
            collection_name=self._vectors.collection_name,
            max_attempts=self.config.vector_outbox_max_attempts,
        )

    def flush_vector_outbox(self) -> tuple[int, int]:
        if self._semantic_outbox_processor is None:
            return 0, 0
        return self._semantic_outbox_processor.process_batch(
            batch_size=self.config.vector_outbox_worker_batch_size,
        )

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[MemoryRecord]:
        _ = kwargs
        query_vector = self._embeddings.embed(query)
        search_limit = max(limit * 3, limit)
        hits = self._vectors.search(
            query_vector=query_vector,
            user_id=self.user_id,
            limit=search_limit,
            session_id=session_id,
        )
        vector_scores = {hit.memory_id: hit.score for hit in hits}
        candidate_ids = list(vector_scores.keys())

        if hasattr(self._store, "list_pending_embedding"):
            pending_facts = self._store.list_pending_embedding(
                self.user_id,
                session_id=session_id,
                limit=self.config.semantic_read_your_writes_limit,
            )
            for fact in pending_facts:
                if fact.id not in candidate_ids:
                    candidate_ids.append(fact.id)
                    vector_scores.setdefault(fact.id, 0.55)

        query_concepts = self._concept_extractor.extract(query, {})
        expanded_scores: dict[str, float] = {}
        if hasattr(self._store, "expand_graph_candidates"):
            expanded_scores = self._store.expand_graph_candidates(
                self.user_id,
                query_concepts,
                max_hops=self.config.semantic_graph_max_hops,
                hop_decay=self.config.semantic_graph_hop_decay,
                limit=self.config.semantic_graph_expansion_limit,
                session_id=session_id,
            )
        for memory_id in expanded_scores:
            if memory_id not in candidate_ids:
                candidate_ids.append(memory_id)

        if not candidate_ids:
            return []

        facts = self._store.get_many(candidate_ids)
        direct_graph_scores = self._store.score_related_memories(
            user_id=self.user_id,
            query_concepts=query_concepts,
            memory_ids=candidate_ids,
        )

        scored: list[tuple[float, MemoryRecord]] = []
        for fact in facts:
            record = _semantic_fact_to_record(fact)
            vector_score = vector_scores.get(fact.id, 0.0)
            graph_score = max(
                direct_graph_scores.get(fact.id, 0.0),
                expanded_scores.get(fact.id, 0.0),
            )
            importance_weight = 0.8 + (record.importance * 0.4)
            final_score = (vector_score * 0.7 + graph_score * 0.3) * importance_weight
            scored.append((final_score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def remove(self, memory_id: str) -> None:
        if not hasattr(self._store, "delete_memory_with_outbox"):
            raise ValueError("语义记忆 store 需实现 delete_memory_with_outbox")
        self._store.delete_memory_with_outbox(
            memory_id=memory_id,
            user_id=self.user_id,
            embedding_model=self.config.embed_model_name,
            collection_name=self._vectors.collection_name,
            max_attempts=self.config.vector_outbox_max_attempts,
        )

    def list_for_user(
        self,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        facts = self._store.list_by_user(
            self.user_id,
            limit=limit,
            session_id=session_id,
        )
        return [_semantic_fact_to_record(fact) for fact in facts]

    def remove_all_for_user(self, session_id: str | None = None) -> int:
        facts = self._store.list_by_user(
            self.user_id,
            limit=10_000,
            session_id=session_id,
        )
        for fact in facts:
            self.remove(fact.id)
        return len(facts)

    def count_for_user(self, session_id: str | None = None) -> int:
        return self._store.count_by_user(self.user_id, session_id=session_id)


def _semantic_fact_to_record(fact: Any) -> MemoryRecord:
    metadata = dict(fact.metadata)
    metadata.setdefault("concepts", list(fact.concepts))
    return MemoryRecord(
        id=fact.id,
        content=fact.content,
        memory_type="semantic",
        importance=fact.importance,
        metadata=metadata,
    )
