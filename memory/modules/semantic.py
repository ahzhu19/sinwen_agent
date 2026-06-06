"""语义记忆模块。"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ..concept_extractor import ConceptExtractor, create_concept_extractor, extract_concepts
from ..config import MemoryConfig
from ..semantic_outbox_processor import SemanticOutboxProcessor
from .base import MemoryRecord
from .semantic_retrieve import retrieve_with_rrf


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
        return retrieve_with_rrf(
            config=self.config,
            store=self._store,
            vectors=self._vectors,
            embeddings=self._embeddings,
            concept_extractor=self._concept_extractor,
            user_id=self.user_id,
            query=query,
            limit=limit,
            session_id=session_id,
        )

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
