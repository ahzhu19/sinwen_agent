"""工作记忆模块。"""

from __future__ import annotations

import copy

import math
import time
from collections import Counter
from typing import Any

from .base import BaseMemory, MemoryRecord
from ..tokenizer import tokenize as _tokenize_text


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
        enriched_metadata = copy.deepcopy(metadata)
        enriched_metadata.setdefault("created_at", now)
        enriched_metadata.setdefault("expires_at", now + self.config.working_memory_ttl_seconds)
        memory_id = super().add(content, importance, enriched_metadata)
        self._enforce_capacity()
        return memory_id

    def list_recent(self, session_id: str) -> list[MemoryRecord]:
        self.cleanup_expired()
        records = [
            record
            for record in self._store.list_records(memory_type=self.memory_type)
            if record.metadata.get("session_id") == session_id
        ]
        return sorted(records, key=lambda record: record.metadata.get("created_at", 0))

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._store.get(memory_id)

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
            for record in self._store.list_records(memory_type=self.memory_type)
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
        for record in self._store.list_records(memory_type=self.memory_type):
            if record.metadata.get("session_id") == session_id:
                self._store.remove(record.id)

    def remove(self, memory_id: str) -> None:
        self._store.remove(memory_id)

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        return self._store.update(
            memory_id,
            content=content,
            importance=importance,
            metadata=metadata,
        )

    def cleanup_expired(self) -> None:
        now = time.time()
        for record in self._store.list_records(memory_type=self.memory_type):
            if record.metadata.get("expires_at", now) <= now:
                self._store.remove(record.id)

    def _enforce_capacity(self) -> None:
        records = self._store.list_records(memory_type=self.memory_type)
        records.sort(key=lambda record: (record.importance, record.metadata.get("created_at", 0)))
        overflow = len(records) - self.config.working_memory_capacity
        for record in records[:max(0, overflow)]:
            self._store.remove(record.id)

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
        query_stripped = query.strip().lower()
        content_lower = content.lower()
        if not query_stripped:
            return 0.0
        if query_stripped in content_lower:
            return 1.0

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        content_tokens = set(self._tokenize(content))
        matched = sum(1 for token in query_tokens if token in content_tokens)
        token_score = matched / len(query_tokens)

        query_grams = set(self._character_bigrams(query_stripped))
        content_grams = set(self._character_bigrams(content_lower))
        if not query_grams:
            return token_score
        gram_score = len(query_grams & content_grams) / len(query_grams)
        return max(token_score, gram_score)

    def _calculate_time_decay(self, created_at: float) -> float:
        age_seconds = max(0.0, time.time() - created_at)
        ttl = max(1, self.config.working_memory_ttl_seconds)
        return max(0.1, 1.0 - (age_seconds / ttl))

    def _tokenize(self, text: str) -> list[str]:
        return _tokenize_text(text)

    def _character_bigrams(self, text: str) -> list[str]:
        compact = "".join(ch for ch in text if not ch.isspace())
        if len(compact) < 2:
            return [compact] if compact else []
        return [compact[index : index + 2] for index in range(len(compact) - 1)]

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
