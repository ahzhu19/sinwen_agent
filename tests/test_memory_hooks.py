"""Memory hook helpers and MemoryService runtime methods."""

from __future__ import annotations

from memory.hooks import (
    DEFAULT_HOOK_SEARCH_MEMORY_TYPES,
    MemoryHookConfig,
    build_interaction_content,
    format_retrieved_context,
)
from memory.service import MemoryService
from tests.memory_fakes import FakeMemoryManager


def test_format_retrieved_context_skips_empty_results() -> None:
    assert format_retrieved_context({}) == ""
    assert format_retrieved_context({"working": []}) == ""


def test_format_retrieved_context_renders_sections() -> None:
    context = format_retrieved_context(
        {
            "working": [{"content": "用户喜欢 Python"}],
            "semantic": [{"content": "Python 是编程语言"}],
        }
    )

    assert "## 相关记忆（自动检索）" in context
    assert "### 工作记忆" in context
    assert "- 用户喜欢 Python" in context
    assert "### 语义记忆" in context


def test_build_interaction_content() -> None:
    content = build_interaction_content("你好", "你好，我能帮你什么？")

    assert content.startswith("用户: 你好")
    assert "助手: 你好，我能帮你什么？" in content


def test_default_hook_search_memory_types() -> None:
    config = MemoryHookConfig()
    assert config.search_memory_types == list(DEFAULT_HOOK_SEARCH_MEMORY_TYPES)


def test_memory_service_retrieve_context_skips_disabled_types() -> None:
    manager = FakeMemoryManager(memory_id="ctx_2")
    service = MemoryService(manager=manager, memory_types=["working"])

    service.retrieve_context("Python", memory_types=["working", "episodic"])

    assert len(manager.searches) == 1
    assert manager.searches[0]["memory_type"] == "working"


def test_memory_service_retrieve_context_delegates_search() -> None:
    manager = FakeMemoryManager(memory_id="ctx_1")
    service = MemoryService(manager=manager, memory_types=["working", "semantic"])

    context = service.retrieve_context(
        "Python",
        session_id="session_1",
        limit_per_type=2,
    )

    assert "fake" in context
    assert len(manager.searches) == 2
    assert manager.searches[0]["memory_type"] == "working"
    assert manager.searches[1]["memory_type"] == "semantic"


def test_memory_service_record_interaction_writes_working_memory() -> None:
    manager = FakeMemoryManager(memory_id="turn_1")
    service = MemoryService(manager=manager)

    memory_id = service.record_interaction(
        "记住我喜欢深色主题",
        "好的，已记住。",
        session_id="session_1",
    )

    assert memory_id == "turn_1"
    assert manager.added[0]["memory_type"] == "working"
    assert "用户: 记住我喜欢深色主题" in manager.added[0]["content"]
    assert manager.added[0]["metadata"]["source"] == "agent_hook"
    assert manager.added[0]["metadata"]["session_id"] == "session_1"
