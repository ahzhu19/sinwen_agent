"""情景记忆模块。"""

from __future__ import annotations

import time
from typing import Any

from ..config import MemoryConfig
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
    ) -> None:
        self.config = config
        self.user_id = user_id
        self._store = episodic_store
        self._vectors = vector_store
        self._embeddings = embedding_provider

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and not session_id:
            session_id = None

        event = self._store.insert(
            user_id=self.user_id,
            content=content,
            importance=importance,
            metadata=dict(metadata),
            session_id=session_id,
        )
        vector = self._embeddings.embed(content)
        self._vectors.upsert(
            memory_id=event.id,
            vector=vector,
            user_id=self.user_id,
            session_id=session_id,
        )
        return event.id

    def get(self, memory_id: str) -> MemoryRecord | None:
        event = self._store.get(memory_id)
        if event is None:
            return None
        return _episodic_event_to_record(event)

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
        return [_episodic_event_to_record(event) for event in events]

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

        events = self._store.get_many([hit.memory_id for hit in hits])
        score_by_id = {hit.memory_id: hit.score for hit in hits}

        scored: list[tuple[float, MemoryRecord]] = []
        for event in events:
            record = _episodic_event_to_record(event)
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
        self._store.delete(memory_id)
        self._vectors.delete(memory_id)


def _episodic_event_to_record(event: Any) -> MemoryRecord:
    meta = dict(event.metadata)
    meta.setdefault("session_id", event.session_id)
    meta.setdefault("occurred_at", event.occurred_at.timestamp())
    meta.setdefault("created_at", event.created_at.timestamp())
    meta.setdefault("sequence_no", event.sequence_no)
    return MemoryRecord(
        id=event.id,
        content=event.content,
        memory_type="episodic",
        importance=event.importance,
        metadata=meta,
    )
