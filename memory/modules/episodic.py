"""情景记忆模块。"""

from __future__ import annotations

import copy

import time
from typing import Any

from ..config import MemoryConfig
from ..records import episodic_event_to_record
from ..storage.postgres_store import PostgresEpisodicMemoryStore
from ..storage.vector_outbox import (
    VectorOutbox,
    VectorWriteError,
    upsert_vector_with_outbox,
)
from ..vector_outbox_processor import VectorOutboxProcessor
from ..orphan_detection import detect_orphan_vectors
from .base import MemoryRecord


class EpisodicMemory:
    """情景记忆：PostgreSQL 结构化存储 + Milvus 向量检索。"""

    memory_type = "episodic"

    def __init__(
        self,
        config: MemoryConfig,
        user_id: str,
        episodic_store: Any,
        vector_store: Any,
        embedding_provider: Any,
        vector_outbox: VectorOutbox | None = None,
        pg_vector_outbox: Any | None = None,
        outbox_processor: VectorOutboxProcessor | None = None,
    ) -> None:
        self.config = config
        self.user_id = user_id
        self._store = episodic_store
        self._vectors = vector_store
        self._embeddings = embedding_provider
        self._vector_outbox = vector_outbox
        self._pg_vector_outbox = pg_vector_outbox
        self._outbox_processor = outbox_processor

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and not session_id:
            session_id = None

        vector = self._embeddings.embed(content)

        if self._use_pg_outbox():
            assert isinstance(self._store, PostgresEpisodicMemoryStore)
            event = self._store.insert_with_vector_outbox(
                user_id=self.user_id,
                content=content,
                importance=importance,
                metadata=copy.deepcopy(metadata),
                session_id=session_id,
                vector=vector,
                collection_name=self._vectors.collection_name,
                max_attempts=self.config.vector_outbox_max_attempts,
                embedding_model=self.config.embed_model_name,
            )
            return event.id

        event = self._store.insert(
            user_id=self.user_id,
            content=content,
            importance=importance,
            metadata=copy.deepcopy(metadata),
            session_id=session_id,
        )
        queued = upsert_vector_with_outbox(
            outbox=self._vector_outbox,
            kind="episodic",
            vector_store=self._vectors,
            memory_id=event.id,
            vector=vector,
            user_id=self.user_id,
            session_id=session_id,
        )
        if not queued and self._vector_outbox is None:
            raise VectorWriteError(
                f"情景记忆 {event.id} 已写入 Postgres，但 Milvus 向量写入失败"
            )
        return event.id

    def update(
        self,
        memory_id: str,
        *,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        existing = self._store.get(memory_id)
        if existing is None or existing.user_id != self.user_id:
            raise KeyError(f"未找到记忆: {memory_id}")

        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and not session_id:
            session_id = None

        vector = self._embeddings.embed(content)

        if self._use_pg_outbox():
            assert isinstance(self._store, PostgresEpisodicMemoryStore)
            self._store.update_with_vector_outbox(
                memory_id=memory_id,
                user_id=self.user_id,
                content=content,
                importance=importance,
                metadata=copy.deepcopy(metadata),
                session_id=session_id,
                vector=vector,
                collection_name=self._vectors.collection_name,
                max_attempts=self.config.vector_outbox_max_attempts,
                embedding_model=self.config.embed_model_name,
            )
            return memory_id

        self._store.update(
            memory_id=memory_id,
            user_id=self.user_id,
            content=content,
            importance=importance,
            metadata=copy.deepcopy(metadata),
            session_id=session_id,
        )
        queued = upsert_vector_with_outbox(
            outbox=self._vector_outbox,
            kind="episodic",
            vector_store=self._vectors,
            memory_id=memory_id,
            vector=vector,
            user_id=self.user_id,
            session_id=session_id,
        )
        if not queued and self._vector_outbox is None:
            raise VectorWriteError(
                f"情景记忆 {memory_id} 已更新 Postgres，但 Milvus 向量写入失败"
            )
        return memory_id

    def flush_vector_outbox(self) -> tuple[int, int]:
        if self._outbox_processor is not None:
            return self._outbox_processor.process_batch(
                batch_size=self.config.vector_outbox_worker_batch_size,
                memory_kind="episodic",
            )
        if self._vector_outbox is None:
            return 0, 0
        return self._vector_outbox.flush(self._vectors, "episodic")

    def get(self, memory_id: str) -> MemoryRecord | None:
        event = self._store.get(memory_id)
        if event is None:
            return None
        return episodic_event_to_record(event)

    def list_timeline(
        self,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        events = self._store.list_timeline(
            user_id=self.user_id,
            session_id=session_id,
            limit=limit,
        )
        return [episodic_event_to_record(event) for event in events]

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[MemoryRecord]:
        _ = kwargs
        query_vector = self._embeddings.embed(query)
        hits = self._vectors.search(
            query_vector=query_vector,
            user_id=self.user_id,
            limit=limit,
            session_id=session_id,
        )
        if not hits:
            return []

        hit_ids = [hit.memory_id for hit in hits]
        events = self._store.get_many(hit_ids)
        detect_orphan_vectors(hit_ids, [e.id for e in events], memory_kind="episodic")
        score_by_id = {hit.memory_id: hit.score for hit in hits}

        scored: list[tuple[float, MemoryRecord]] = []
        for event in events:
            record = episodic_event_to_record(event)
            vector_score = score_by_id.get(event.id, 0.0)
            occurred_at = record.metadata.get(
                "occurred_at",
                record.metadata.get("created_at", time.time()),
            )
            recency_score = self._calculate_time_recency(float(occurred_at))
            importance_weight = 0.8 + (record.importance * 0.4)
            base_relevance = vector_score * 0.8 + recency_score * 0.2
            final_score = base_relevance * importance_weight
            scored.append((final_score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def _calculate_time_recency(self, occurred_at: float) -> float:
        """时间近因性：越近的事件分数越高，范围约 [0.1, 1.0]。"""
        age_seconds = max(0.0, time.time() - occurred_at)
        window = max(1, self.config.episodic_memory_recency_seconds)
        return max(0.1, 1.0 - (age_seconds / window))

    def remove(self, memory_id: str) -> None:
        if self._pg_vector_outbox is not None:
            self._pg_vector_outbox.enqueue_delete(
                memory_kind="episodic",
                memory_id=memory_id,
                user_id=self.user_id,
                session_id=None,
                collection_name=self._vectors.collection_name,
                max_attempts=self.config.vector_outbox_max_attempts,
            )
        self._store.delete(memory_id)
        try:
            self._vectors.delete(memory_id)
        except Exception:
            if self._pg_vector_outbox is None and self._vector_outbox is None:
                raise

    def _use_pg_outbox(self) -> bool:
        return (
            self._pg_vector_outbox is not None
            and isinstance(self._store, PostgresEpisodicMemoryStore)
            and self.config.enable_persistent_vector_outbox
        )
