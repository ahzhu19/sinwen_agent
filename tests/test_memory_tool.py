"""MemoryTool tests."""

import pytest

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.service import MemoryService
from tests.memory_fakes import FakeMemoryManager, FailingMemoryManager
from tools.agent_registry import create_agent_tool_registry
from tools.builtin.memory_tool import MemoryTool


class RecordingMemoryTool(MemoryTool):
    def __init__(self, user_id: str = "user123") -> None:
        super().__init__(user_id=user_id, memory_types=["working"])

    def _add_memory(self, **kwargs) -> str:
        return f"add:{kwargs}"

    def _search_memory(self, **kwargs) -> str:
        return f"search:{kwargs}"

    def _get_summary(self, **kwargs) -> str:
        return f"summary:{kwargs}"

    def _get_stats(self, **kwargs) -> str:
        return f"stats:{kwargs}"

    def _update_memory(self, **kwargs) -> str:
        return f"update:{kwargs}"

    def _remove_memory(self, **kwargs) -> str:
        return f"remove:{kwargs}"

    def _forget_memory(self, **kwargs) -> str:
        return f"forget:{kwargs}"

    def _consolidate_memory(self, **kwargs) -> str:
        return f"consolidate:{kwargs}"

    def _clear_all_memories(self, **kwargs) -> str:
        return f"clear_all:{kwargs}"


@pytest.mark.parametrize(
    ("action", "expected_prefix"),
    [
        ("add", "add"),
        ("search", "search"),
        ("summary", "summary"),
        ("stats", "stats"),
        ("update", "update"),
        ("remove", "remove"),
        ("forget", "forget"),
        ("consolidate", "consolidate"),
        ("clear_all", "clear_all"),
    ],
)
def test_memory_tool_execute_dispatches_supported_actions(
    action: str,
    expected_prefix: str,
) -> None:
    tool = RecordingMemoryTool(user_id="user123")

    result = tool.execute(action, content="hello", memory_type="semantic")

    assert result.startswith(f"{expected_prefix}:")
    assert "'content': 'hello'" in result
    assert "'memory_type': 'semantic'" in result


def test_memory_tool_run_delegates_to_execute() -> None:
    tool = RecordingMemoryTool(user_id="user123")

    result = tool.run(action="add", content="hello")

    assert result.startswith("add:")


def test_memory_tool_execute_rejects_unknown_action() -> None:
    tool = RecordingMemoryTool(user_id="user123")

    result = tool.execute("unknown")

    assert "不支持的记忆操作" in result
    assert "unknown" in result


def test_add_memory_delegates_to_memory_manager_with_session_metadata() -> None:
    manager = FakeMemoryManager("abcdef123456")
    tool = MemoryTool(user_id="user123", memory_manager=manager)

    result = tool.execute(
        "add",
        content="用户张三是一名 Python 开发者",
        memory_type="semantic",
        importance=0.8,
        source="chat",
    )

    assert result == "✅ 记忆已添加 (ID: abcdef12...)"
    assert tool.current_session_id is not None
    assert len(manager.calls) == 1
    call = manager.calls[0]
    assert call["content"] == "用户张三是一名 Python 开发者"
    assert call["memory_type"] == "semantic"
    assert call["importance"] == 0.8
    assert call["metadata"]["source"] == "chat"
    assert call["metadata"]["session_id"] == tool.current_session_id
    assert "timestamp" in call["metadata"]


def test_add_perceptual_memory_infers_modality_from_file_path() -> None:
    manager = FakeMemoryManager()
    tool = MemoryTool(user_id="user123", memory_manager=manager, session_id="session_1")

    result = tool.execute(
        "add",
        content="用户上传了一张架构图",
        memory_type="perceptual",
        file_path="/tmp/architecture.png",
    )

    assert "✅ 记忆已添加" in result
    metadata = manager.calls[0]["metadata"]
    assert metadata["session_id"] == "session_1"
    assert metadata["modality"] == "image"
    assert metadata["raw_data"] == "/tmp/architecture.png"


def test_add_perceptual_memory_keeps_explicit_modality() -> None:
    manager = FakeMemoryManager()
    tool = MemoryTool(user_id="user123", memory_manager=manager)

    tool.execute(
        "add",
        content="用户上传了语音片段",
        memory_type="perceptual",
        file_path="/tmp/voice.bin",
        modality="audio",
    )

    assert manager.calls[0]["metadata"]["modality"] == "audio"


def test_add_memory_auto_creates_manager_when_none_provided() -> None:
    tool = MemoryTool(user_id="user123")

    result = tool.execute("add", content="hello")

    assert "✅ 记忆已添加" in result
    assert isinstance(tool.memory_manager, MemoryManager)


def test_add_memory_returns_clear_error_when_manager_fails() -> None:
    tool = MemoryTool(user_id="user123", memory_manager=FailingMemoryManager())

    result = tool.execute("add", content="hello")

    assert result == "❌ 添加记忆失败: database unavailable"


def test_memory_tool_accepts_memory_service() -> None:
    manager = FakeMemoryManager("service_mem_123")
    service = MemoryService(manager=manager)
    tool = MemoryTool(user_id="user123", memory_service=service)

    result = tool.execute(
        "add",
        content="用户喜欢 Python",
        memory_type="working",
        importance=0.8,
    )

    assert "service_" in result
    assert manager.added[0]["content"] == "用户喜欢 Python"


def test_agent_tool_registry_accepts_memory_service() -> None:
    service = MemoryService(manager=FakeMemoryManager("registry_mem_123"))
    registry = create_agent_tool_registry(
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        memory_service=service,
    )

    tool = registry._tools["memory"]
    result = tool.execute("add", content="hello", memory_type="working")

    assert "registry" in result


def test_memory_tool_creates_manager_from_config_and_memory_types() -> None:
    config = MemoryConfig(working_memory_capacity=20)

    tool = MemoryTool(
        user_id="user123",
        memory_config=config,
        memory_types=["working"],
    )

    assert tool.memory_config is config
    assert tool.memory_types == ["working"]
    assert isinstance(tool.memory_manager, MemoryManager)
    assert tool.memory_manager.user_id == "user123"
    assert tool.memory_manager.config is config
    assert tool.memory_manager.enable_working is True
    assert tool.memory_manager.enable_episodic is False
    assert tool.memory_manager.enable_semantic is False
    assert tool.memory_manager.enable_perceptual is False


def test_memory_tool_keeps_injected_manager() -> None:
    manager = FakeMemoryManager()

    tool = MemoryTool(user_id="user123", memory_manager=manager)

    assert tool.memory_manager is manager
    assert tool.memory_service.manager is manager
    assert tool.memory_types == ["working"]


def test_memory_tool_rejects_unknown_memory_type_during_initialization() -> None:
    with pytest.raises(ValueError, match="不支持的记忆类型"):
        MemoryTool(user_id="user123", memory_types=["working", "invalid"])
