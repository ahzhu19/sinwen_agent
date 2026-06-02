"""情景记忆测试用 fake 后端。"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from memory.episodic_store import EpisodicEvent
from memory.vector_store import VectorSearchHit


class FakeEmbeddingProvider:
    def __init__(self, vector_size: int = 8) -> None:
        self._vector_size = vector_size
        self.calls: list[str] = []

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_vector(text, self._vector_size) for text in texts]


class FakeEpisodicStore:
    def __init__(self) -> None:
        self.events: dict[str, EpisodicEvent] = {}
        self._sequence = 0

    def ensure_schema(self) -> None:
        return None

    def insert(
        self,
        user_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        session_id: str | None = None,
        occurred_at: datetime | None = None,
        memory_id: str | None = None,
    ) -> EpisodicEvent:
        self._sequence += 1
        event_id = memory_id or str(uuid4())
        now = datetime.now(timezone.utc)
        event = EpisodicEvent(
            id=event_id,
            user_id=user_id,
            session_id=session_id,
            content=content,
            importance=importance,
            occurred_at=occurred_at or now,
            created_at=now,
            sequence_no=self._sequence,
            metadata=dict(metadata),
        )
        self.events[event_id] = event
        return event

    def get(self, memory_id: str) -> EpisodicEvent | None:
        return self.events.get(memory_id)

    def get_many(self, memory_ids: list[str]) -> list[EpisodicEvent]:
        return [self.events[mid] for mid in memory_ids if mid in self.events]

    def list_timeline(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[EpisodicEvent]:
        events = [
            event
            for event in self.events.values()
            if event.user_id == user_id
            and (session_id is None or event.session_id == session_id)
        ]
        events.sort(key=lambda event: event.sequence_no)
        return events[-limit:]

    def delete(self, memory_id: str) -> None:
        self.events.pop(memory_id, None)


class FakeVectorStore:
    def __init__(self) -> None:
        self.collection_name = "fake_episodic_vectors"
        self.records: dict[str, dict[str, Any]] = {}

    def ensure_collection(self, vector_size: int) -> None:
        _ = vector_size

    def upsert(
        self,
        memory_id: str,
        vector: list[float],
        user_id: str,
        session_id: str | None,
    ) -> None:
        self.records[memory_id] = {
            "vector": vector,
            "user_id": user_id,
            "session_id": session_id or "",
        }

    def search(
        self,
        query_vector: list[float],
        user_id: str,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[VectorSearchHit]:
        hits: list[VectorSearchHit] = []
        for memory_id, record in self.records.items():
            if record["user_id"] != user_id:
                continue
            if session_id and record["session_id"] != session_id:
                continue
            score = _cosine(query_vector, record["vector"])
            hits.append(
                VectorSearchHit(
                    memory_id=memory_id,
                    score=score,
                    user_id=user_id,
                    session_id=record["session_id"] or None,
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def delete(self, memory_id: str) -> None:
        self.records.pop(memory_id, None)


def _hash_to_vector(text: str, size: int) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [((digest[index % len(digest)] / 255.0) * 2) - 1 for index in range(size)]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
