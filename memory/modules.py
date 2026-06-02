"""不同类型记忆模块。"""

from __future__ import annotations

import time
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .config import MemoryConfig


@dataclass
class MemoryRecord:
    id: str
    content: str
    memory_type: str
    importance: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryStore:
    """临时内存 store，后续可替换为数据库实现。"""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._records_by_type: dict[str, dict[str, MemoryRecord]] = {}

    def add(self, record: MemoryRecord) -> str:
        self._records[record.id] = record
        self._records_by_type.setdefault(record.memory_type, {})[record.id] = record
        return record.id

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def list_records(self, memory_type: str | None = None) -> list[MemoryRecord]:
        if memory_type is not None:
            return list(self._records_by_type.get(memory_type, {}).values())
        return list(self._records.values())

    def remove(self, memory_id: str) -> None:
        record = self._records.pop(memory_id, None)
        if record is None:
            return

        typed_records = self._records_by_type.get(record.memory_type)
        if typed_records is None:
            return

        typed_records.pop(memory_id, None)
        if not typed_records:
            self._records_by_type.pop(record.memory_type, None)


class BaseMemory:
    memory_type: str

    def __init__(self, config: MemoryConfig, store: InMemoryStore) -> None:
        self.config = config
        self.store = store

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        memory_id = str(uuid4())
        return self.store.add(
            MemoryRecord(
                id=memory_id,
                content=content,
                memory_type=self.memory_type,
                importance=importance,
                metadata=metadata,
            )
        )


class WorkingMemory(BaseMemory):
    memory_type = "working"

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        self.cleanup_expired()
        now = time.time()
        enriched_metadata = dict(metadata)
        enriched_metadata.setdefault("created_at", now)
        enriched_metadata.setdefault("expires_at", now + self.config.working_memory_ttl_seconds)
        memory_id = super().add(content, importance, enriched_metadata)
        self._enforce_capacity()
        return memory_id

    def list_recent(self, session_id: str) -> list[MemoryRecord]:
        self.cleanup_expired()
        records = [
            record
            for record in self.store.list_records(memory_type=self.memory_type)
            if record.metadata.get("session_id") == session_id
        ]
        return sorted(records, key=lambda record: record.metadata.get("created_at", 0))

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[MemoryRecord]:
        _ = kwargs
        self.cleanup_expired()
        candidates = [
            record
            for record in self.store.list_records(memory_type=self.memory_type)
            if session_id is None or record.metadata.get("session_id") == session_id
        ]
        vector_scores = self._try_tfidf_search(query, candidates)

        scored_records: list[tuple[float, MemoryRecord]] = []
        for record in candidates:
            vector_score = vector_scores.get(record.id, 0.0)
            keyword_score = self._calculate_keyword_score(query, record.content)
            base_relevance = (
                vector_score * 0.7 + keyword_score * 0.3
                if vector_score > 0
                else keyword_score
            )
            time_decay = self._calculate_time_decay(record.metadata.get("created_at", time.time()))
            importance_weight = 0.8 + (record.importance * 0.4)
            final_score = base_relevance * time_decay * importance_weight
            if final_score > 0:
                scored_records.append((final_score, record))

        scored_records.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored_records[:limit]]

    def clear_session(self, session_id: str) -> None:
        for record in self.store.list_records(memory_type=self.memory_type):
            if record.metadata.get("session_id") == session_id:
                self.store.remove(record.id)

    def cleanup_expired(self) -> None:
        now = time.time()
        for record in self.store.list_records(memory_type=self.memory_type):
            if record.metadata.get("expires_at", now) <= now:
                self.store.remove(record.id)

    def _enforce_capacity(self) -> None:
        records = self.store.list_records(memory_type=self.memory_type)
        records.sort(key=lambda record: (record.importance, record.metadata.get("created_at", 0)))
        overflow = len(records) - self.config.working_memory_capacity
        for record in records[:max(0, overflow)]:
            self.store.remove(record.id)

    def _try_tfidf_search(
        self,
        query: str,
        memories: list[MemoryRecord],
    ) -> dict[str, float]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {}

        documents = {record.id: self._tokenize(record.content) for record in memories}
        document_count = len(documents)
        if document_count == 0:
            return {}

        document_frequency: Counter[str] = Counter()
        for tokens in documents.values():
            document_frequency.update(set(tokens))

        query_vector = self._tfidf_vector(query_tokens, document_frequency, document_count)
        scores: dict[str, float] = {}
        for memory_id, tokens in documents.items():
            document_vector = self._tfidf_vector(tokens, document_frequency, document_count)
            score = self._cosine_similarity(query_vector, document_vector)
            if score > 0:
                scores[memory_id] = score
        return scores

    def _calculate_keyword_score(self, query: str, content: str) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0
        content_tokens = set(self._tokenize(content))
        matched = sum(1 for token in query_tokens if token in content_tokens)
        return matched / len(query_tokens)

    def _calculate_time_decay(self, created_at: float) -> float:
        age_seconds = max(0.0, time.time() - created_at)
        ttl = max(1, self.config.working_memory_ttl_seconds)
        return max(0.1, 1.0 - (age_seconds / ttl))

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())

    def _tfidf_vector(
        self,
        tokens: list[str],
        document_frequency: Counter[str],
        document_count: int,
    ) -> dict[str, float]:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vector: dict[str, float] = {}
        for token, count in counts.items():
            tf = count / total
            idf = math.log((document_count + 1) / (document_frequency[token] + 1)) + 1
            vector[token] = tf * idf
        return vector

    def _cosine_similarity(
        self,
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        common_tokens = set(left) & set(right)
        numerator = sum(left[token] * right[token] for token in common_tokens)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)


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


class PerceptualMemory(BaseMemory):
    memory_type = "perceptual"


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
