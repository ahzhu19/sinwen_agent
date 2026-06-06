"""Milvus 向量 outbox 补偿测试。"""

from __future__ import annotations

from memory.config import MemoryConfig
from memory.modules import EpisodicMemory
from memory.storage.vector_outbox import VectorOutbox
from tests.episodic_fakes import FakeEmbeddingProvider, FakeEpisodicStore, FakeVectorStore


class FlakyVectorStore(FakeVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.failures_before_success = 1
        self.upsert_calls = 0

    def upsert(
        self,
        memory_id: str,
        vector: list[float],
        user_id: str,
        session_id: str | None,
    ) -> None:
        self.upsert_calls += 1
        if self.failures_before_success > 0:
            self.failures_before_success -= 1
            raise RuntimeError("milvus unavailable")
        super().upsert(memory_id, vector, user_id, session_id)


def test_episodic_add_enqueues_failed_vector_write() -> None:
    outbox = VectorOutbox(max_attempts=3)
    store = FakeEpisodicStore()
    vectors = FlakyVectorStore()
    memory = EpisodicMemory(
        config=MemoryConfig(enable_vector_outbox=True),
        user_id="user123",
        episodic_store=store,
        vector_store=vectors,
        embedding_provider=FakeEmbeddingProvider(vector_size=8),
        vector_outbox=outbox,
    )

    memory_id = memory.add("事件内容", 0.7, {"session_id": "s1"})
    assert memory_id in store.events
    assert memory_id not in vectors.records
    assert outbox.pending_count() == 1

    succeeded, failed = memory.flush_vector_outbox()
    assert succeeded == 1
    assert failed == 0
    assert memory_id in vectors.records
    assert outbox.pending_count() == 0


def test_episodic_flush_vector_outbox_before_retrieve() -> None:
    """检索前需显式 flush（MemoryManager.search_memory 在 poll 时负责）。"""
    outbox = VectorOutbox(max_attempts=3)
    store = FakeEpisodicStore()
    vectors = FlakyVectorStore()
    memory = EpisodicMemory(
        config=MemoryConfig(),
        user_id="user123",
        episodic_store=store,
        vector_store=vectors,
        embedding_provider=FakeEmbeddingProvider(vector_size=8),
        vector_outbox=outbox,
    )
    memory.add("PostgreSQL 迁移完成", 0.8, {"session_id": "s1"})
    vectors.failures_before_success = 0
    memory.flush_vector_outbox()

    results = memory.retrieve("PostgreSQL", limit=3)
    assert len(results) >= 1
    assert vectors.records
