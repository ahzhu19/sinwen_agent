"""语义记忆测试用 fake 后端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from memory.storage.neo4j_store import SemanticFact
from memory.storage.semantic_outbox_types import SemanticMemorySyncState, SemanticOutboxEvent
from tests.episodic_fakes import FakeEmbeddingProvider, FakeVectorStore


@dataclass
class _FakeOutboxEvent:
    event_id: str
    memory_id: str
    user_id: str
    version: int
    operation: str
    status: str = "pending"
    attempts: int = 0
    max_attempts: int = 5
    last_error: str | None = None
    embedding_model: str = ""
    collection_name: str = ""
    session_id: str | None = None


class FakeSemanticStore:
    def __init__(self) -> None:
        self.facts: dict[str, SemanticFact] = {}
        self.graph_scores: dict[str, float] = {}
        self.expanded_scores: dict[str, float] = {}
        self.outbox_events: dict[str, _FakeOutboxEvent] = {}

    def ensure_schema(self) -> None:
        return None

    def write_memory_with_outbox(
        self,
        *,
        user_id: str,
        memory_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        concepts: list[str],
        operation: str,
        embedding_model: str,
        collection_name: str,
        max_attempts: int = 5,
    ) -> SemanticFact:
        existing = self.facts.get(memory_id)
        version = 1 if existing is None else existing.version + 1
        session_id = metadata.get("session_id")
        session_value = session_id if isinstance(session_id, str) else None
        fact = SemanticFact(
            id=memory_id,
            user_id=user_id,
            content=content,
            importance=importance,
            concepts=concepts,
            metadata=dict(metadata),
            version=version,
            embedding_version=0,
            embedding_status="pending",
            embedding_model=embedding_model,
            deleted=False,
        )
        self.facts[memory_id] = fact
        event = _FakeOutboxEvent(
            event_id=str(uuid4()),
            memory_id=memory_id,
            user_id=user_id,
            version=version,
            operation=operation,
            embedding_model=embedding_model,
            collection_name=collection_name,
            max_attempts=max_attempts,
            session_id=session_value,
        )
        self.outbox_events[event.event_id] = event
        return fact

    def delete_memory_with_outbox(
        self,
        *,
        memory_id: str,
        user_id: str,
        embedding_model: str,
        collection_name: str,
        max_attempts: int = 5,
    ) -> None:
        existing = self.facts.get(memory_id)
        if existing is None:
            return
        version = existing.version + 1
        fact = SemanticFact(
            id=existing.id,
            user_id=existing.user_id,
            content=existing.content,
            importance=existing.importance,
            concepts=list(existing.concepts),
            metadata=dict(existing.metadata),
            version=version,
            embedding_version=existing.embedding_version,
            embedding_status="pending",
            embedding_model=embedding_model,
            deleted=True,
        )
        self.facts[memory_id] = fact
        event = _FakeOutboxEvent(
            event_id=str(uuid4()),
            memory_id=memory_id,
            user_id=user_id,
            version=version,
            operation="delete",
            embedding_model=embedding_model,
            collection_name=collection_name,
            max_attempts=max_attempts,
        )
        self.outbox_events[event.event_id] = event

    def claim_pending_outbox_events(self, *, batch_size: int = 20) -> list[SemanticOutboxEvent]:
        pending = [
            event
            for event in self.outbox_events.values()
            if event.status == "pending" and event.attempts < event.max_attempts
        ][:batch_size]
        for event in pending:
            event.status = "processing"
            event.attempts += 1
        return [_fake_event_to_outbox(event) for event in pending]

    def mark_outbox_done(self, event_id: str) -> None:
        event = self.outbox_events.get(event_id)
        if event is not None:
            event.status = "done"

    def mark_outbox_failed(self, event_id: str, error: str, *, max_attempts: int) -> None:
        event = self.outbox_events.get(event_id)
        if event is None:
            return
        event.last_error = error
        event.status = "dead" if event.attempts >= max_attempts else "pending"

    def mark_outbox_superseded(self, event_id: str) -> None:
        event = self.outbox_events.get(event_id)
        if event is not None:
            event.status = "superseded"

    def get_memory_sync_state(self, memory_id: str) -> SemanticMemorySyncState | None:
        fact = self.facts.get(memory_id)
        if fact is None:
            return None
        session_id = fact.metadata.get("session_id")
        session_value = session_id if isinstance(session_id, str) else None
        return SemanticMemorySyncState(
            id=fact.id,
            user_id=fact.user_id,
            content=fact.content,
            importance=fact.importance,
            concepts=list(fact.concepts),
            metadata=dict(fact.metadata),
            version=fact.version,
            embedding_version=fact.embedding_version,
            embedding_status=fact.embedding_status,
            embedding_model=fact.embedding_model,
            deleted=fact.deleted,
            session_id=session_value,
        )

    def update_embedding_sync_state(
        self,
        memory_id: str,
        *,
        embedding_version: int,
        embedding_status: str,
        embedding_model: str,
    ) -> None:
        fact = self.facts.get(memory_id)
        if fact is None:
            return
        self.facts[memory_id] = SemanticFact(
            id=fact.id,
            user_id=fact.user_id,
            content=fact.content,
            importance=fact.importance,
            concepts=list(fact.concepts),
            metadata=dict(fact.metadata),
            version=fact.version,
            embedding_version=embedding_version,
            embedding_status=embedding_status,
            embedding_model=embedding_model,
            deleted=fact.deleted,
        )

    def list_pending_embedding(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[SemanticFact]:
        facts = [
            fact
            for fact in self.facts.values()
            if fact.user_id == user_id
            and fact.embedding_status == "pending"
            and not fact.deleted
        ]
        if session_id is not None:
            facts = [fact for fact in facts if fact.metadata.get("session_id") == session_id]
        return facts[:limit]

    def pending_outbox_count(self) -> int:
        return sum(
            1
            for event in self.outbox_events.values()
            if event.status in {"pending", "processing"}
        )

    def get_many(self, memory_ids: list[str]) -> list[SemanticFact]:
        return [
            self.facts[memory_id]
            for memory_id in memory_ids
            if memory_id in self.facts and not self.facts[memory_id].deleted
        ]

    def compute_graph_relevance(
        self,
        user_id: str,
        query_concepts: list[str],
        *,
        max_hops: int = 2,
        hop_decay: float = 0.65,
        relation_weights: dict[str, float] | None = None,
        session_id: str | None = None,
    ) -> dict[str, float]:
        _ = user_id, query_concepts, max_hops, hop_decay, relation_weights, session_id
        scores: dict[str, float] = {}
        for memory_id, score in self.graph_scores.items():
            scores[memory_id] = max(scores.get(memory_id, 0.0), score)
        for memory_id, score in self.expanded_scores.items():
            scores[memory_id] = max(scores.get(memory_id, 0.0), score)
        return scores

    def list_by_user(
        self,
        user_id: str,
        limit: int = 10_000,
        session_id: str | None = None,
    ) -> list[SemanticFact]:
        facts = [
            fact
            for fact in self.facts.values()
            if fact.user_id == user_id and not fact.deleted
        ]
        if session_id is not None:
            facts = [fact for fact in facts if fact.metadata.get("session_id") == session_id]
        return facts[:limit]

    def count_by_user(self, user_id: str, session_id: str | None = None) -> int:
        return len(self.list_by_user(user_id, session_id=session_id))

    def delete(self, memory_id: str) -> None:
        self.facts.pop(memory_id, None)


def _fake_event_to_outbox(event: _FakeOutboxEvent) -> SemanticOutboxEvent:
    return SemanticOutboxEvent(
        event_id=event.event_id,
        memory_id=event.memory_id,
        user_id=event.user_id,
        version=event.version,
        operation=event.operation,
        status=event.status,
        attempts=event.attempts,
        max_attempts=event.max_attempts,
        last_error=event.last_error,
        embedding_model=event.embedding_model,
        collection_name=event.collection_name,
        session_id=event.session_id,
    )


@dataclass
class SemanticFakeBundle:
    store: FakeSemanticStore
    vectors: FakeVectorStore
    embeddings: FakeEmbeddingProvider


def create_semantic_bundle(vector_size: int = 8) -> SemanticFakeBundle:
    return SemanticFakeBundle(
        store=FakeSemanticStore(),
        vectors=FakeVectorStore(),
        embeddings=FakeEmbeddingProvider(vector_size=vector_size),
    )


def create_semantic_memory_with_outbox(
    bundle: SemanticFakeBundle,
    *,
    user_id: str = "user123",
    concept_extractor: Any = None,
    config: Any = None,
) -> tuple[Any, Any]:
    from memory.config import MemoryConfig
    from memory.modules.semantic import SemanticMemory
    from memory.semantic_outbox_processor import SemanticOutboxProcessor

    memory_config = config or MemoryConfig()
    processor = SemanticOutboxProcessor(
        memory_config,
        bundle.store,
        embedding_provider=bundle.embeddings,
        vector_store=bundle.vectors,
    )
    memory = SemanticMemory(
        config=memory_config,
        user_id=user_id,
        semantic_store=bundle.store,
        vector_store=bundle.vectors,
        embedding_provider=bundle.embeddings,
        concept_extractor=concept_extractor,
        semantic_outbox_processor=processor,
    )
    return memory, processor
