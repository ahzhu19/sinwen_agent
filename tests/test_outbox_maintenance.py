"""Outbox 维护与 P1 可靠性测试。"""

from __future__ import annotations

from memory.config import MemoryConfig
from memory.outbox_maintenance import run_memory_outbox_maintenance
from memory.storage.postgres_outbox_store import VectorOutboxRecord
from memory.vector_outbox_processor import VectorOutboxProcessor
from tests.episodic_fakes import FakeEpisodicStore, FakeVectorStore
from tests.test_postgres_outbox_processor import FakePostgresOutbox


class FakeEpisodicWithIndex(FakeEpisodicStore):
    def __init__(self) -> None:
        super().__init__()
        self.indexed: set[str] = set()

    def mark_vector_indexed(self, memory_id: str) -> None:
        self.indexed.add(memory_id)

    def count_unindexed_vectors(self) -> int:
        return 0


def _set_status(outbox: FakePostgresOutbox, entry_id: int, status: str, *, attempts: int = 0) -> None:
    entry = outbox._entries[entry_id]
    outbox._entries[entry_id] = VectorOutboxRecord(
        id=entry.id,
        memory_kind=entry.memory_kind,
        memory_id=entry.memory_id,
        user_id=entry.user_id,
        session_id=entry.session_id,
        collection_name=entry.collection_name,
        op=entry.op,
        vector=entry.vector,
        status=status,
        attempts=attempts,
        max_attempts=entry.max_attempts,
    )


def test_pg_outbox_reclaim_stale_processing() -> None:
    outbox = FakePostgresOutbox()
    outbox.enqueue_upsert(
        memory_kind="episodic",
        memory_id="mem-1",
        user_id="user1",
        session_id=None,
        collection_name="episodic_vectors",
        vector=[0.1],
        max_attempts=3,
    )
    _set_status(outbox, 1, "processing")
    reclaimed = outbox.reclaim_stale_processing(timeout_seconds=0)
    assert reclaimed == 1
    assert outbox._entries[1].status == "pending"


def test_pg_outbox_replay_dead() -> None:
    outbox = FakePostgresOutbox()
    outbox.enqueue_upsert(
        memory_kind="episodic",
        memory_id="mem-dead",
        user_id="user1",
        session_id=None,
        collection_name="episodic_vectors",
        vector=[0.1],
        max_attempts=3,
    )
    _set_status(outbox, 1, "dead", attempts=5)
    replayed = outbox.replay_dead(batch_size=10)
    assert replayed == 1
    assert outbox._entries[1].status == "pending"
    assert outbox._entries[1].attempts == 0


def test_vector_processor_marks_episodic_indexed_on_success() -> None:
    outbox = FakePostgresOutbox()
    vectors = FakeVectorStore()
    episodic = FakeEpisodicWithIndex()
    outbox.enqueue_upsert(
        memory_kind="episodic",
        memory_id="mem-1",
        user_id="user1",
        session_id="s1",
        collection_name=vectors.collection_name,
        vector=[0.1, 0.2],
        max_attempts=3,
    )
    processor = VectorOutboxProcessor(
        MemoryConfig(),
        outbox,
        vector_stores_by_collection={vectors.collection_name: vectors},
        episodic_store=episodic,
    )
    ok_count, fail_count = processor.process_batch(batch_size=10, reclaim_stale=False)
    assert ok_count == 1
    assert fail_count == 0
    assert "mem-1" in episodic.indexed


def test_run_memory_outbox_maintenance_counts() -> None:
    pg = FakePostgresOutbox()
    pg.enqueue_upsert(
        memory_kind="episodic",
        memory_id="mem-stale",
        user_id="user1",
        session_id=None,
        collection_name="episodic_vectors",
        vector=[0.1],
        max_attempts=3,
    )
    _set_status(pg, 1, "processing")
    results = run_memory_outbox_maintenance(
        MemoryConfig(vector_outbox_processing_timeout_seconds=0),
        pg_outbox=pg,
        reclaim_stale=True,
        replay_dead=False,
        reconcile_semantic=False,
    )
    assert results["pg_reclaimed"] == 1
