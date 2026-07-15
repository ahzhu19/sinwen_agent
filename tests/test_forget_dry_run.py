"""forget dry-run 预览功能测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.modules.base import MemoryRecord
from memory.service import MemoryService
from tests.episodic_fakes import FakeEmbeddingProvider, FakeEpisodicStore, FakeVectorStore
from tests.memory_fakes import FakeMemoryManager
from tools.builtin.memory_tool import MemoryTool


class FakeEpisodicStoreWithForget(FakeEpisodicStore):
    def list_for_forget(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        limit: int = 1000,
    ):
        return self.list_timeline(user_id, session_id=session_id, limit=limit)


def _working_manager() -> MemoryManager:
    return MemoryManager(
        config=MemoryConfig(),
        user_id="dry_user",
        enable_episodic=False,
        enable_semantic=False,
    )


# ---------- Manager 层 ----------

def test_dry_run_working_returns_preview_without_deleting() -> None:
    manager = _working_manager()
    manager.add_memory("低价值笔记", "working", 0.1, {"session_id": "s1"})
    manager.add_memory("重要结论", "working", 0.9, {"session_id": "s1"})

    preview = manager.forget_memories("working", importance_threshold=0.5, dry_run=True)

    assert isinstance(preview, list)
    assert len(preview) == 1
    assert isinstance(preview[0], MemoryRecord)
    assert preview[0].content == "低价值笔记"
    stats = manager.memory_stats(session_id="s1")
    assert stats["counts"]["working"] == 2


def test_dry_run_returns_empty_when_nothing_matches() -> None:
    manager = _working_manager()
    manager.add_memory("重要结论", "working", 0.9, {"session_id": "s1"})

    preview = manager.forget_memories("working", importance_threshold=0.5, dry_run=True)

    assert preview == []
    assert manager.memory_stats(session_id="s1")["counts"]["working"] == 1


def test_dry_run_then_real_forget_removes_same_records() -> None:
    manager = _working_manager()
    manager.add_memory("低A", "working", 0.1, {"session_id": "s1"})
    manager.add_memory("低B", "working", 0.2, {"session_id": "s1"})
    manager.add_memory("高C", "working", 0.9, {"session_id": "s1"})

    preview = manager.forget_memories("working", importance_threshold=0.5, dry_run=True)
    assert len(preview) == 2
    assert manager.memory_stats(session_id="s1")["counts"]["working"] == 3

    removed = manager.forget_memories("working", importance_threshold=0.5)
    assert removed == 2
    assert manager.memory_stats(session_id="s1")["counts"]["working"] == 1


def test_dry_run_episodic_returns_preview_without_deleting() -> None:
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
    low_old_id = store.insert("user1", "旧低价值", 0.1, {}, occurred_at=old).id
    high_old_id = store.insert("user1", "旧高价值", 0.9, {}, occurred_at=old).id

    preview = manager.forget_memories(
        "episodic",
        strategy="importance_ttl",
        importance_threshold=0.2,
        older_than_days=30,
        dry_run=True,
    )

    assert isinstance(preview, list)
    assert len(preview) == 1
    assert isinstance(preview[0], MemoryRecord)
    assert preview[0].content == "旧低价值"
    assert store.get(low_old_id) is not None
    assert store.get(high_old_id) is not None


def test_dry_run_respects_limit() -> None:
    manager = _working_manager()
    for i in range(5):
        manager.add_memory(f"低{i}", "working", 0.1, {"session_id": "s1"})

    preview = manager.forget_memories(
        "working", importance_threshold=0.5, limit=2, dry_run=True
    )

    assert len(preview) == 2


# ---------- Service 层 ----------

def test_service_forget_dry_run_delegates_to_manager() -> None:
    fake = FakeMemoryManager()
    fake.forget_dry_run_records = [
        {"id": "mem-1", "content": "将删除的笔记"},
        {"id": "mem-2", "content": "另一条"},
    ]
    service = MemoryService(manager=fake)

    result = service.forget("working", dry_run=True)

    assert isinstance(result, list)
    assert len(result) == 2


def test_service_forget_without_dry_run_returns_int() -> None:
    fake = FakeMemoryManager()
    fake.forgotten_count = 3
    service = MemoryService(manager=fake)

    result = service.forget("working")

    assert result == 3


# ---------- Tool 层 ----------

def _working_tool() -> MemoryTool:
    return MemoryTool(user_id="dry_tool_user", memory_types=["working"])


def test_tool_forget_dry_run_previews_without_deleting() -> None:
    tool = _working_tool()
    tool.execute("add", content="临时笔记", importance=0.1)
    tool.execute("add", content="重要结论", importance=0.9)

    preview = tool.execute("forget", importance_threshold=0.5, dry_run=True)

    assert "预览" in preview
    assert "dry-run 未实际删除" in preview
    assert "临时笔记" in preview
    assert "重要结论" not in preview
    stats = tool.execute("stats")
    assert "working: 2 条" in stats


def test_tool_forget_dry_run_empty_preview() -> None:
    tool = _working_tool()
    tool.execute("add", content="重要结论", importance=0.9)

    preview = tool.execute("forget", importance_threshold=0.5, dry_run=True)

    assert "预览" in preview
    assert "将遗忘 0 条" in preview


def test_tool_forget_without_dry_run_still_deletes() -> None:
    tool = _working_tool()
    tool.execute("add", content="临时笔记", importance=0.1)
    tool.execute("add", content="重要结论", importance=0.9)

    result = tool.execute("forget", importance_threshold=0.5)

    assert "已遗忘 1 条" in result
    stats = tool.execute("stats")
    assert "working: 1 条" in stats
