"""MemoryManager productization tests."""

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from tools.builtin.memory_tool import MemoryTool


def _working_episodic_tool() -> MemoryTool:
    config = MemoryConfig(working_memory_capacity=50)
    manager = MemoryManager(
        config=config,
        user_id="test_user",
        enable_working=True,
        enable_episodic=False,
        enable_semantic=False,
        enable_perceptual=False,
    )
    return MemoryTool(
        user_id="test_user",
        session_id="session_ops",
        memory_manager=manager,
        memory_types=["working"],
    )


def test_memory_tool_stats_summary_remove_forget() -> None:
    tool = _working_episodic_tool()
    tool.execute("add", content="临时笔记", importance=0.1)
    tool.execute("add", content="重要结论", importance=0.9)

    stats = tool.execute("stats")
    assert "working: 2 条" in stats

    summary = tool.execute("summary", limit=5)
    assert "重要结论" in summary

    forget = tool.execute("forget", importance_threshold=0.5)
    assert "已遗忘 1 条" in forget

    stats_after = tool.execute("stats")
    assert "working: 1 条" in stats_after


def test_memory_tool_update_and_clear_all() -> None:
    tool = _working_episodic_tool()
    add_result = tool.execute("add", content="旧内容", importance=0.6)
    assert "记忆已添加" in add_result

    working = tool.memory_manager.memory_modules["working"]
    record = working.store.list_records(memory_type="working")[0]
    update = tool.execute(
        "update",
        memory_id=record.id,
        memory_type="working",
        content="新内容",
    )
    assert "记忆已更新" in update

    updated = working.store.get(record.id)
    assert updated is not None
    assert updated.content == "新内容"

    clear = tool.execute("clear_all")
    assert "working: 1 条" in clear
    assert tool.execute("stats") == (
        "用户 test_user 记忆统计\n"
        "已启用类型: working\n"
        "- working: 0 条"
    )


def test_memory_tool_consolidate_requires_episodic() -> None:
    tool = _working_episodic_tool()
    tool.execute("add", content="待整合", importance=0.8)
    result = tool.execute("consolidate")
    assert "未启用 episodic" in result
