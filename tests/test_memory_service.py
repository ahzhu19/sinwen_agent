"""MemoryService tests."""

from __future__ import annotations

from memory.service import MemoryService
from tests.memory_fakes import FakeMemoryManager


def test_memory_service_delegates_crud_operations() -> None:
    manager = FakeMemoryManager(memory_id="memory_123")
    service = MemoryService(manager=manager)

    memory_id = service.add(
        content="记住用户喜欢 Python",
        memory_type="working",
        importance=0.8,
        metadata={"session_id": "session_1"},
    )
    results = service.search(
        query="Python",
        memory_type="working",
        limit=3,
        session_id="session_1",
    )
    updated_id = service.update(
        "memory_123",
        "working",
        content="用户喜欢 Python 和 Neo4j",
        importance=0.9,
        metadata={"topic": "preference"},
    )
    service.remove("memory_123", "working")

    assert memory_id == "memory_123"
    assert updated_id == "memory_123"
    assert results
    assert manager.added[0]["content"] == "记住用户喜欢 Python"
    assert manager.searches[0]["query"] == "Python"
    assert manager.updated[0]["content"] == "用户喜欢 Python 和 Neo4j"
    assert manager.removed == [("memory_123", "working")]


def test_memory_service_delegates_lifecycle_operations() -> None:
    manager = FakeMemoryManager()
    manager.forgotten_count = 2
    manager.consolidated_ids = ["ep_1", "ep_2"]
    manager.cleared = {"working": 3}
    service = MemoryService(manager=manager)

    forgotten = service.forget(
        "working",
        strategy="importance_ttl",
        session_id="session_1",
        importance_threshold=0.2,
        older_than_days=7,
        limit=5,
    )
    consolidated = service.consolidate("session_1", importance_threshold=0.7)
    cleared = service.clear(memory_type="working", session_id="session_1")
    stats = service.stats(session_id="session_1")
    summary = service.summary(session_id="session_1", limit_per_type=2)

    assert forgotten == 2
    assert consolidated == ["ep_1", "ep_2"]
    assert cleared == {"working": 3}
    assert stats["user_id"] == manager.stats_user_id
    assert "working" in summary
