"""EpisodicMemory tests with fake PostgreSQL/Milvus backends."""

from __future__ import annotations

import time

import pytest

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.modules import EpisodicMemory, MemoryRecord
from tests.episodic_fakes import (
    FakeEmbeddingProvider,
    FakeEpisodicStore,
    FakeVectorStore,
)


@pytest.fixture
def episodic_bundle() -> tuple[FakeEpisodicStore, FakeVectorStore, FakeEmbeddingProvider]:
    return FakeEpisodicStore(), FakeVectorStore(), FakeEmbeddingProvider(vector_size=8)


@pytest.fixture
def episodic_memory(episodic_bundle: tuple) -> EpisodicMemory:
    store, vectors, embeddings = episodic_bundle
    return EpisodicMemory(
        config=MemoryConfig(),
        user_id="user123",
        episodic_store=store,
        vector_store=vectors,
        embedding_provider=embeddings,
    )


def test_episodic_memory_add_writes_postgres_and_milvus(
    episodic_memory: EpisodicMemory,
    episodic_bundle: tuple,
) -> None:
    store, vectors, embeddings = episodic_bundle

    memory_id = episodic_memory.add(
        content="用户昨天完成了 PostgreSQL 迁移",
        importance=0.8,
        metadata={"session_id": "session_1", "source": "chat"},
    )

    assert memory_id in store.events
    assert memory_id in vectors.records
    assert embeddings.calls == ["用户昨天完成了 PostgreSQL 迁移"]
    event = store.events[memory_id]
    assert event.user_id == "user123"
    assert event.session_id == "session_1"
    assert event.content.startswith("用户昨天")


def test_episodic_memory_retrieve_returns_semantically_related_events(
    episodic_memory: EpisodicMemory,
) -> None:
    first_id = episodic_memory.add(
        "用户正在学习 Python 机器学习",
        0.6,
        {"session_id": "session_1"},
    )
    episodic_memory.add(
        "用户也喜欢 React 前端开发",
        0.5,
        {"session_id": "session_1"},
    )

    results = episodic_memory.retrieve(
        "Python 机器学习",
        limit=2,
        session_id="session_1",
    )

    assert len(results) >= 1
    assert any(record.id == first_id for record in results)
    assert all(isinstance(record, MemoryRecord) for record in results)


def test_episodic_memory_retrieve_uses_vector_recency_and_importance_formula(
    episodic_memory: EpisodicMemory,
    episodic_bundle: tuple,
) -> None:
    store, vectors, _ = episodic_bundle
    now = time.time()

    fresh_id = episodic_memory.add(
        "用户正在学习 Python 机器学习",
        0.6,
        {"session_id": "session_1"},
    )
    old_id = episodic_memory.add(
        "用户正在学习 Python 机器学习旧记录",
        1.0,
        {"session_id": "session_1"},
    )

    store.events[old_id].metadata["occurred_at"] = now - 25 * 24 * 3600
    vectors.records[old_id]["vector"] = list(vectors.records[fresh_id]["vector"])
    vectors.records[old_id]["session_id"] = "session_1"

    results = episodic_memory.retrieve(
        "Python 机器学习",
        limit=2,
        session_id="session_1",
    )

    assert [record.id for record in results] == [fresh_id, old_id]


def test_episodic_memory_list_timeline_preserves_sequence(
    episodic_memory: EpisodicMemory,
) -> None:
    episodic_memory.add("第一条事件", 0.5, {"session_id": "session_1"})
    episodic_memory.add("第二条事件", 0.5, {"session_id": "session_1"})
    episodic_memory.add("其他会话", 0.5, {"session_id": "session_2"})

    timeline = episodic_memory.list_timeline(session_id="session_1")

    assert [record.content for record in timeline] == ["第一条事件", "第二条事件"]


def test_episodic_memory_remove_deletes_both_stores(
    episodic_memory: EpisodicMemory,
    episodic_bundle: tuple,
) -> None:
    store, vectors, _ = episodic_bundle
    memory_id = episodic_memory.add("待删除事件", 0.5, {"session_id": "session_1"})

    episodic_memory.remove(memory_id)

    assert store.get(memory_id) is None
    assert memory_id not in vectors.records


def test_episodic_memory_update_preserves_id_and_reindexes_vector(
    episodic_memory: EpisodicMemory,
    episodic_bundle: tuple,
) -> None:
    store, vectors, embeddings = episodic_bundle

    memory_id = episodic_memory.add(
        "用户完成了 PostgreSQL 迁移",
        0.7,
        {"session_id": "session_1"},
    )
    original_sequence = store.events[memory_id].sequence_no

    updated_id = episodic_memory.update(
        memory_id,
        content="用户完成了 PostgreSQL 与 Milvus 迁移",
        importance=0.9,
        metadata={"session_id": "session_1", "source": "chat"},
    )

    assert updated_id == memory_id
    assert len(store.events) == 1
    event = store.events[memory_id]
    assert event.content == "用户完成了 PostgreSQL 与 Milvus 迁移"
    assert event.importance == 0.9
    assert event.sequence_no == original_sequence
    assert memory_id in vectors.records
    assert embeddings.calls[-1] == "用户完成了 PostgreSQL 与 Milvus 迁移"


def test_episodic_memory_update_raises_when_missing(episodic_memory: EpisodicMemory) -> None:
    with pytest.raises(KeyError, match="未找到记忆"):
        episodic_memory.update(
            "missing-id",
            content="内容",
            importance=0.5,
            metadata={},
        )


def test_memory_manager_episodic_uses_injected_backends(episodic_bundle: tuple) -> None:
    store, vectors, embeddings = episodic_bundle
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=False,
        enable_episodic=True,
        enable_semantic=False,
        episodic_store=store,
        vector_store=vectors,
        embedding_provider=embeddings,
    )

    memory_id = manager.add_memory(
        content="情景记忆事件",
        memory_type="episodic",
        importance=0.7,
        metadata={"session_id": "session_1"},
    )

    assert memory_id in store.events
    results = manager.search_memory(
        query="情景记忆",
        memory_type="episodic",
        limit=3,
        session_id="session_1",
    )
    assert len(results) >= 1


def test_memory_tool_search_episodic_memory() -> None:
    from tools.builtin.memory_tool import MemoryTool

    store = FakeEpisodicStore()
    vectors = FakeVectorStore()
    embeddings = FakeEmbeddingProvider(vector_size=8)
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=False,
        enable_episodic=True,
        enable_semantic=False,
        episodic_store=store,
        vector_store=vectors,
        embedding_provider=embeddings,
    )
    tool = MemoryTool(
        user_id="user123",
        session_id="session_1",
        memory_manager=manager,
        memory_types=["episodic"],
    )

    tool.execute(
        "add",
        content="用户完成了 Milvus 向量库接入",
        memory_type="episodic",
    )
    result = tool.execute(
        "search",
        query="Milvus 向量",
        memory_type="episodic",
    )

    assert "找到" in result
    assert "Milvus" in result
