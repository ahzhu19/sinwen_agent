"""Postgres outbox processor tests（内存 fake，无需真实数据库）。"""

from __future__ import annotations

from memory.config import MemoryConfig
from memory.storage.postgres_outbox_store import VectorOutboxRecord
from memory.vector_outbox_processor import VectorOutboxProcessor
from tests.episodic_fakes import FakeVectorStore


class FakePostgresOutbox:
    def __init__(self) -> None:
        self._entries: dict[int, VectorOutboxRecord] = {}
        self._seq = 0

    def ensure_schema(self) -> None:
        return None

    def pending_count(self, *, memory_kind: str | None = None) -> int:
        _ = memory_kind
        return sum(1 for entry in self._entries.values() if entry.status == "pending")

    def enqueue_upsert(
        self,
        *,
        memory_kind: str,
        memory_id: str,
        user_id: str,
        session_id: str | None,
        collection_name: str,
        vector: list[float],
        max_attempts: int,
    ) -> None:
        self._seq += 1
        self._entries[self._seq] = VectorOutboxRecord(
            id=self._seq,
            memory_kind=memory_kind,
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
            collection_name=collection_name,
            op="upsert",
            vector=vector,
            status="pending",
            attempts=0,
            max_attempts=max_attempts,
        )

    def enqueue_delete(
        self,
        *,
        memory_kind: str,
        memory_id: str,
        user_id: str,
        session_id: str | None,
        collection_name: str,
        max_attempts: int,
    ) -> None:
        self._seq += 1
        self._entries[self._seq] = VectorOutboxRecord(
            id=self._seq,
            memory_kind=memory_kind,
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
            collection_name=collection_name,
            op="delete",
            vector=None,
            status="pending",
            attempts=0,
            max_attempts=max_attempts,
        )

    def claim_pending(self, *, batch_size: int = 20) -> list[VectorOutboxRecord]:
        pending = [
            entry
            for entry in self._entries.values()
            if entry.status == "pending"
        ][:batch_size]
        claimed: list[VectorOutboxRecord] = []
        for entry in pending:
            claimed.append(
                VectorOutboxRecord(
                    id=entry.id,
                    memory_kind=entry.memory_kind,
                    memory_id=entry.memory_id,
                    user_id=entry.user_id,
                    session_id=entry.session_id,
                    collection_name=entry.collection_name,
                    op=entry.op,
                    vector=entry.vector,
                    status="processing",
                    attempts=entry.attempts,
                    max_attempts=entry.max_attempts,
                )
            )
            self._entries[entry.id] = claimed[-1]
        return claimed

    def mark_done(self, entry_id: int) -> None:
        entry = self._entries[entry_id]
        self._entries[entry_id] = VectorOutboxRecord(
            id=entry.id,
            memory_kind=entry.memory_kind,
            memory_id=entry.memory_id,
            user_id=entry.user_id,
            session_id=entry.session_id,
            collection_name=entry.collection_name,
            op=entry.op,
            vector=entry.vector,
            status="done",
            attempts=entry.attempts,
            max_attempts=entry.max_attempts,
        )

    def mark_failed(self, entry_id: int, error: str, *, max_attempts: int) -> None:
        _ = error, max_attempts
        entry = self._entries[entry_id]
        self._entries[entry_id] = VectorOutboxRecord(
            id=entry.id,
            memory_kind=entry.memory_kind,
            memory_id=entry.memory_id,
            user_id=entry.user_id,
            session_id=entry.session_id,
            collection_name=entry.collection_name,
            op=entry.op,
            vector=entry.vector,
            status="pending",
            attempts=entry.attempts + 1,
            max_attempts=entry.max_attempts,
        )

    def reclaim_stale_processing(self, *, timeout_seconds: int) -> int:
        _ = timeout_seconds
        reclaimed = 0
        for entry_id, entry in list(self._entries.items()):
            if entry.status != "processing":
                continue
            self._entries[entry_id] = VectorOutboxRecord(
                id=entry.id,
                memory_kind=entry.memory_kind,
                memory_id=entry.memory_id,
                user_id=entry.user_id,
                session_id=entry.session_id,
                collection_name=entry.collection_name,
                op=entry.op,
                vector=entry.vector,
                status="pending",
                attempts=entry.attempts,
                max_attempts=entry.max_attempts,
            )
            reclaimed += 1
        return reclaimed

    def replay_dead(self, *, batch_size: int = 20, memory_kind: str | None = None) -> int:
        replayed = 0
        for entry_id, entry in list(self._entries.items()):
            if replayed >= batch_size:
                break
            if memory_kind is not None and entry.memory_kind != memory_kind:
                continue
            if entry.status != "dead":
                continue
            self._entries[entry_id] = VectorOutboxRecord(
                id=entry.id,
                memory_kind=entry.memory_kind,
                memory_id=entry.memory_id,
                user_id=entry.user_id,
                session_id=entry.session_id,
                collection_name=entry.collection_name,
                op=entry.op,
                vector=entry.vector,
                status="pending",
                attempts=0,
                max_attempts=entry.max_attempts,
            )
            replayed += 1
        return replayed


def test_processor_upserts_vector_to_milvus() -> None:
    vectors = FakeVectorStore()
    vectors.collection_name = "episodic_vectors"
    outbox = FakePostgresOutbox()
    outbox.enqueue_upsert(
        memory_kind="episodic",
        memory_id="mem-1",
        user_id="user1",
        session_id="s1",
        collection_name="episodic_vectors",
        vector=[0.1, 0.2, 0.3],
        max_attempts=3,
    )

    processor = VectorOutboxProcessor(
        MemoryConfig(enable_persistent_vector_outbox=True),
        outbox,
        vector_stores_by_collection={"episodic_vectors": vectors},
    )
    succeeded, failed = processor.process_batch(batch_size=10)

    assert succeeded == 1
    assert failed == 0
    assert "mem-1" in vectors.records


def test_processor_delete_removes_vector() -> None:
    vectors = FakeVectorStore()
    vectors.collection_name = "semantic_vectors"
    vectors.upsert("mem-del", [0.1, 0.2], "user1", "s1")

    outbox = FakePostgresOutbox()
    outbox.enqueue_delete(
        memory_kind="perceptual",
        memory_id="mem-del",
        user_id="user1",
        session_id="s1",
        collection_name="semantic_vectors",
        max_attempts=3,
    )

    processor = VectorOutboxProcessor(
        MemoryConfig(),
        outbox,
        vector_stores_by_collection={"semantic_vectors": vectors},
    )
    succeeded, failed = processor.process_batch(batch_size=10)

    assert succeeded == 1
    assert failed == 0
    assert "mem-del" not in vectors.records
