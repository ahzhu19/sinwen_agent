"""Episodic forget 测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from tests.episodic_fakes import FakeEmbeddingProvider, FakeEpisodicStore, FakeVectorStore


class FakeEpisodicStoreWithForget(FakeEpisodicStore):
    def list_for_forget(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        limit: int = 1000,
    ):
        return self.list_timeline(user_id, session_id=session_id, limit=limit)


def test_forget_episodic_by_importance_and_ttl() -> None:
    store = FakeEpisodicStoreWithForget()
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user1",
        enable_working=False,
        enable_semantic=False,
        episodic_store=store,
        vector_store=FakeVectorStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    old = datetime.now(timezone.utc) - timedelta(days=40)
    recent = datetime.now(timezone.utc) - timedelta(days=1)
    low_old = store.insert("user1", "旧低价值", 0.1, {}, occurred_at=old)
    low_recent = store.insert("user1", "新低价值", 0.1, {}, occurred_at=recent)
    high_old = store.insert("user1", "旧高价值", 0.9, {}, occurred_at=old)

    removed = manager.forget_memories(
        "episodic",
        strategy="importance_ttl",
        importance_threshold=0.2,
        older_than_days=30,
    )
    assert removed == 1
    assert store.get(low_old.id) is None
    assert store.get(low_recent.id) is not None
    assert store.get(high_old.id) is not None
