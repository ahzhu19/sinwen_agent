"""Agent runtime memory hook integration tests."""

from __future__ import annotations

from typing import Any

from agents.react_agent import ReActAgent
from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM, LLMMessages
from memory.hooks import DEFAULT_HOOK_SEARCH_MEMORY_TYPES, MemoryHookConfig
from memory.service import MemoryService
from tests.memory_fakes import FakeMemoryManager
from tools.registry import ToolRegistry


class CapturingLLM(BaseLLM):
    def __init__(self, response: str = "已结合记忆回答") -> None:
        self.model = "capturing"
        self.client = None  # type: ignore[assignment]
        self.response = response
        self.last_messages: LLMMessages = []

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = temperature, kwargs
        self.last_messages = messages
        return self.response


class CapturingReActLLM(BaseLLM):
    def __init__(self) -> None:
        self.model = "capturing-react"
        self.client = None  # type: ignore[assignment]
        self.last_messages: LLMMessages = []
        self.calls = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = temperature, kwargs
        self.calls += 1
        self.last_messages = messages
        return 'Thought: 完成。\nAction: Finish\nAction Input: 已结合记忆回答'


def _system_content(messages: LLMMessages) -> str:
    for message in messages:
        if message.get("role") == "system":
            return str(message.get("content", ""))
    return ""


def test_simple_agent_injects_retrieved_context_without_memory_tool() -> None:
    manager = FakeMemoryManager(memory_id="hook_mem_1")
    service = MemoryService(manager=manager, memory_types=["working"])
    llm = CapturingLLM()

    agent = SimpleAgent(
        name="hook-agent",
        llm=llm,
        system_prompt="你是助手。",
        memory_service=service,
        memory_hooks=MemoryHookConfig(session_id="session_hook"),
    )

    answer = agent.run("用户喜欢什么？")

    assert answer == "已结合记忆回答"
    assert "## 相关记忆（自动检索）" in _system_content(llm.last_messages)
    assert len(manager.searches) == 1
    assert manager.searches[0]["memory_type"] == "working"
    assert len(manager.added) == 1
    assert "用户: 用户喜欢什么？" in manager.added[0]["content"]


def test_react_agent_records_interaction_after_finish() -> None:
    manager = FakeMemoryManager(memory_id="react_hook_1")
    service = MemoryService(manager=manager, memory_types=["working"])
    llm = CapturingReActLLM()

    agent = ReActAgent(
        name="react-hook",
        llm=llm,
        tool_registry=ToolRegistry(),
        memory_service=service,
        memory_hooks=MemoryHookConfig(
            retrieve_before_run=False,
            session_id="react_session",
        ),
    )

    answer = agent.run("帮我总结偏好")

    assert answer == "已结合记忆回答"
    assert len(manager.added) == 1
    assert manager.added[0]["metadata"]["session_id"] == "react_session"


def test_with_agent_tools_can_enable_memory_hooks_only() -> None:
    manager = FakeMemoryManager(memory_id="shared_service")
    service = MemoryService(manager=manager, memory_types=["working"])
    llm = CapturingLLM()

    agent = SimpleAgent.with_agent_tools(
        name="shared-hook",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=False,
        enable_memory_hooks=True,
        memory_service=service,
    )

    assert agent.memory_service is service
    assert agent.memory_hooks is not None
    assert "memory" not in agent.list_tools()

    agent.run("测试 hooks")

    assert len(manager.searches) == 1
    assert manager.searches[0]["memory_type"] == "working"
    assert len(manager.added) == 1


def test_default_hook_search_types_are_working_and_episodic() -> None:
    manager = FakeMemoryManager(memory_id="preset_1")
    service = MemoryService(
        manager=manager,
        memory_types=["working", "episodic", "semantic"],
    )
    llm = CapturingLLM()

    agent = SimpleAgent(
        name="preset-agent",
        llm=llm,
        memory_service=service,
        memory_hooks=MemoryHookConfig(),
    )

    agent.run("查询")

    searched_types = [call["memory_type"] for call in manager.searches]
    assert searched_types == list(DEFAULT_HOOK_SEARCH_MEMORY_TYPES)


def test_enable_memory_enables_hooks_by_default() -> None:
    manager = FakeMemoryManager(memory_id="auto_hook")
    llm = CapturingLLM()

    agent = SimpleAgent.with_agent_tools(
        name="auto-hook",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        memory_service=MemoryService(manager=manager, memory_types=["working"]),
    )

    assert agent.memory_hooks is not None
    assert "memory" in agent.list_tools()

    agent.run("你好")

    assert len(manager.searches) >= 1
    assert len(manager.added) == 1


def test_enable_memory_can_disable_hooks_explicitly() -> None:
    manager = FakeMemoryManager(memory_id="no_hook")
    llm = CapturingLLM()

    agent = SimpleAgent.with_agent_tools(
        name="no-hook",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        enable_memory_hooks=False,
        memory_service=MemoryService(manager=manager, memory_types=["working"]),
    )

    assert agent.memory_hooks is None

    agent.run("你好")

    assert manager.searches == []
    assert manager.added == []
