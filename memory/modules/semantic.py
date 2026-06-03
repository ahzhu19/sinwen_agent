"""语义记忆模块。"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from ..config import MemoryConfig
from .base import MemoryRecord


class SemanticMemory:
    """语义记忆：Neo4j 知识图谱 + Milvus 向量检索。"""

    memory_type = "semantic"

    def __init__(
        self,
        config: MemoryConfig,
        user_id: str,
        semantic_store: Any,
        vector_store: Any,
        embedding_provider: Any,
    ) -> None:
        self.config = config
        self.user_id = user_id
        self._store = semantic_store
        self._vectors = vector_store
        self._embeddings = embedding_provider

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        memory_id = str(uuid4())
        concepts = self._extract_concepts(content, metadata)
        self._store.upsert_memory(
            user_id=self.user_id,
            memory_id=memory_id,
            content=content,
            importance=importance,
            metadata=dict(metadata),
            concepts=concepts,
        )
        vector = self._embeddings.embed(content)
        session_id = metadata.get("session_id")
        self._vectors.upsert(
            memory_id=memory_id,
            vector=vector,
            user_id=self.user_id,
            session_id=session_id if isinstance(session_id, str) else None,
        )
        return memory_id

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

        candidate_ids = [hit.memory_id for hit in hits]
        facts = self._store.get_many(candidate_ids)
        vector_scores = {hit.memory_id: hit.score for hit in hits}
        query_concepts = self._extract_concepts(query, {})
        graph_scores = self._store.score_related_memories(
            user_id=self.user_id,
            query_concepts=query_concepts,
            memory_ids=candidate_ids,
        )

        scored: list[tuple[float, MemoryRecord]] = []
        for fact in facts:
            record = _semantic_fact_to_record(fact)
            vector_score = vector_scores.get(fact.id, 0.0)
            graph_score = graph_scores.get(fact.id, 0.0)
            importance_weight = 0.8 + (record.importance * 0.4)
            final_score = (vector_score * 0.7 + graph_score * 0.3) * importance_weight
            scored.append((final_score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def remove(self, memory_id: str) -> None:
        self._store.delete(memory_id)
        self._vectors.delete(memory_id)

    def _extract_concepts(self, content: str, metadata: dict[str, Any]) -> list[str]:
        raw_concepts = metadata.get("concepts")
        if isinstance(raw_concepts, list):
            concepts = [str(concept).strip() for concept in raw_concepts if str(concept).strip()]
            if concepts:
                return _dedupe(concepts)

        # 第一版不做 LLM 自动抽取，仅用轻量词片段兜底，后续可替换为抽取器。
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", content)
        return _dedupe(tokens[:8])


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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
