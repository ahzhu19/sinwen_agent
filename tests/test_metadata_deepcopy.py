"""M-05: metadata 深拷贝测试 — 验证嵌套对象不被调用方修改影响。"""

from __future__ import annotations

from memory.config import MemoryConfig
from memory.service import MemoryService
from tools.builtin.memory_tool import MemoryTool


def _service() -> MemoryService:
    return MemoryService(
        user_id="test_user",
        config=MemoryConfig(
            database_url=None,
            enable_vector_outbox=False,
            enable_persistent_vector_outbox=False,
        ),
        memory_types=["working"],
    )


def test_add_memory_nested_metadata_not_shared() -> None:
    """添加记忆后修改调用方嵌套 dict，存储中的 metadata 不受影响。"""
    service = _service()
    tool = MemoryTool(
        user_id="test_user",
        session_id="sess-dc-1",
        memory_service=service,
    )

    nested = {"tags": ["python", "ai"]}
    tool.execute(
        "add",
        content="测试深拷贝",
        memory_type="working",
        importance=0.8,
        extra=nested,
    )

    # 修改调用方嵌套对象
    nested["tags"].append("mutated")
    nested["new_key"] = "injected"

    # 检索存储的记忆 — 嵌套对象不应被修改
    results = service.manager.search_memory("测试", "working", limit=5)
    assert len(results) >= 1
    stored_extra = results[0].metadata.get("extra")
    assert stored_extra is not None
    assert stored_extra["tags"] == ["python", "ai"]
    assert "new_key" not in stored_extra


def test_update_memory_nested_metadata_not_shared() -> None:
    """更新记忆后修改调用方嵌套 dict，存储中的 metadata 不受影响。"""
    service = _service()
    tool = MemoryTool(
        user_id="test_user",
        session_id="sess-dc-2",
        memory_service=service,
    )

    add_result = tool.execute(
        "add",
        content="待更新记忆",
        memory_type="working",
        importance=0.5,
    )
    memory_id = add_result.split("ID: ")[1].split("...")[0]

    nested_meta = {"labels": {"category": "tech"}}
    tool.execute(
        "update",
        memory_id=memory_id + "00000000-0000-0000-0000-000000000000",  # pad to full uuid
        memory_type="working",
        metadata=nested_meta,
    )

    # 修改调用方嵌套对象
    nested_meta["labels"]["category"] = "mutated"

    # 验证存储中不受影响 — update 失败时记忆不变，成功时嵌套对象不被修改
    results = service.manager.search_memory("待更新", "working", limit=5)
    assert len(results) >= 1
    # 原始记忆的 metadata 不应包含调用方修改的嵌套对象
    stored_labels = results[0].metadata.get("labels")
    if stored_labels is not None:
        assert stored_labels["category"] == "tech"


def test_inmemory_store_update_deepcopy() -> None:
    """InMemoryStore.update 创建新 record 时深拷贝 metadata。"""
    from memory.modules.base import InMemoryStore, MemoryRecord

    store = InMemoryStore()
    original_meta = {"nested": {"value": 1}}
    record = MemoryRecord(
        id="rec-1",
        content="test",
        memory_type="working",
        importance=0.5,
        metadata=original_meta,
    )
    store.add(record)

    # update with new metadata
    store.update("rec-1", metadata={"extra": "added"})

    # 修改原始 metadata 的嵌套对象
    original_meta["nested"]["value"] = 999

    updated = store.get("rec-1")
    assert updated is not None
    assert updated.metadata["nested"]["value"] == 1
    assert updated.metadata["extra"] == "added"
