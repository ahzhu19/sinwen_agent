"""WorkingMemory tests."""

import time

from memory.config import MemoryConfig
from memory.modules import InMemoryStore, MemoryRecord, WorkingMemory


def test_in_memory_store_lists_records_by_memory_type() -> None:
    store = InMemoryStore()
    working = MemoryRecord("working-1", "工作记忆", "working", 0.5, {})
    semantic = MemoryRecord("semantic-1", "语义记忆", "semantic", 0.5, {})

    store.add(working)
    store.add(semantic)
    store.remove(working.id)

    assert store.list_records(memory_type="working") == []
    assert store.list_records(memory_type="semantic") == [semantic]


def test_working_memory_adds_session_scoped_record() -> None:
    store = InMemoryStore()
    memory = WorkingMemory(MemoryConfig(), store)

    memory_id = memory.add(
        content="当前用户正在讨论记忆系统",
        importance=0.7,
        metadata={"session_id": "session_1"},
    )

    record = store.get(memory_id)
    assert record is not None
    assert record.memory_type == "working"
    assert record.metadata["session_id"] == "session_1"
    assert record.metadata["expires_at"] > record.metadata["created_at"]


def test_working_memory_lists_only_unexpired_session_records() -> None:
    store = InMemoryStore()
    memory = WorkingMemory(MemoryConfig(working_memory_ttl_seconds=10), store)

    first_id = memory.add("第一条", 0.5, {"session_id": "session_1"})
    memory.add("第二条", 0.5, {"session_id": "session_2"})
    store.get(first_id).metadata["expires_at"] = 0  # type: ignore[union-attr]
    memory.add("第三条", 0.5, {"session_id": "session_1"})

    records = memory.list_recent(session_id="session_1")

    assert [record.content for record in records] == ["第三条"]
    assert store.get(first_id) is None


def test_working_memory_enforces_capacity_by_removing_oldest_records() -> None:
    store = InMemoryStore()
    memory = WorkingMemory(
        MemoryConfig(working_memory_capacity=2, working_memory_ttl_seconds=60),
        store,
    )

    first_id = memory.add("第一条", 0.5, {"session_id": "session_1"})
    second_id = memory.add("第二条", 0.5, {"session_id": "session_1"})
    third_id = memory.add("第三条", 0.5, {"session_id": "session_1"})

    assert store.get(first_id) is None
    assert store.get(second_id) is not None
    assert store.get(third_id) is not None
    assert [record.content for record in memory.list_recent("session_1")] == ["第二条", "第三条"]


def test_working_memory_clear_session_removes_only_that_session() -> None:
    store = InMemoryStore()
    memory = WorkingMemory(MemoryConfig(), store)

    first_id = memory.add("第一条", 0.5, {"session_id": "session_1"})
    second_id = memory.add("第二条", 0.5, {"session_id": "session_2"})

    memory.clear_session("session_1")

    assert store.get(first_id) is None
    assert store.get(second_id) is not None


def test_working_memory_capacity_removes_lowest_priority_record() -> None:
    store = InMemoryStore()
    memory = WorkingMemory(
        MemoryConfig(working_memory_capacity=2, working_memory_ttl_seconds=60),
        store,
    )

    high_id = memory.add("高优先级", 0.9, {"session_id": "session_1"})
    low_id = memory.add("低优先级", 0.1, {"session_id": "session_1"})
    newest_id = memory.add("新信息", 0.5, {"session_id": "session_1"})

    assert store.get(low_id) is None
    assert store.get(high_id) is not None
    assert store.get(newest_id) is not None


def test_working_memory_retrieve_uses_hybrid_relevance_importance_and_time_decay() -> None:
    store = InMemoryStore()
    memory = WorkingMemory(MemoryConfig(working_memory_ttl_seconds=3600), store)

    python_id = memory.add("用户正在学习 Python 机器学习", 0.6, {"session_id": "session_1"})
    old_id = memory.add("Python 数据分析旧上下文", 1.0, {"session_id": "session_1"})
    react_id = memory.add("用户也喜欢 React 前端开发", 1.0, {"session_id": "session_1"})
    old_record = store.get(old_id)
    assert old_record is not None
    old_record.metadata["created_at"] = time.time() - 3500

    results = memory.retrieve("Python 机器学习", limit=2, session_id="session_1")

    assert [record.id for record in results] == [python_id, old_id]
    assert react_id not in [record.id for record in results]


def test_working_memory_retrieve_cleans_expired_records() -> None:
    store = InMemoryStore()
    memory = WorkingMemory(MemoryConfig(working_memory_ttl_seconds=60), store)

    expired_id = memory.add("Python 旧信息", 1.0, {"session_id": "session_1"})
    fresh_id = memory.add("Python 新信息", 0.5, {"session_id": "session_1"})
    expired_record = store.get(expired_id)
    assert expired_record is not None
    expired_record.metadata["expires_at"] = 0

    results = memory.retrieve("Python", session_id="session_1")

    assert [record.id for record in results] == [fresh_id]
    assert store.get(expired_id) is None


def test_working_memory_retrieve_matches_chinese_substring_in_content() -> None:
    store = InMemoryStore()
    memory = WorkingMemory(MemoryConfig(working_memory_ttl_seconds=3600), store)

    theme_id = memory.add(
        "用户偏好深色主题界面",
        0.6,
        {"session_id": "session_1"},
    )
    memory.add("无关的浅色背景说明", 0.5, {"session_id": "session_1"})

    results = memory.retrieve("深色主题", limit=5, session_id="session_1")

    assert [record.id for record in results] == [theme_id]
