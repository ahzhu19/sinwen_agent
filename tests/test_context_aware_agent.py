"""ContextAwareAgent integration tests."""

from __future__ import annotations

from typing import Any

from agents.context_aware_agent import ContextAwareAgent
from context import ContextBuilder, ContextConfig
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse
from memory.hooks import MemoryHookConfig
from memory.modules.base import MemoryRecord
from memory.service import MemoryService
from tests.memory_fakes import FakeMemoryManager
from tests.test_rag_tool import FakeRagManager
from tools.builtin.memory_tool import MemoryTool
from tools.builtin.rag_tool import RagTool


class CapturingLLM(BaseLLM):
    def __init__(self, response: str = "已结合上下文回答") -> None:
        self.model = "capturing-context"
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

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        _ = tools, tool_choice, temperature, kwargs
        self.last_messages = messages
        return LLMToolResponse(content=self.response, tool_calls=None)


class SearchableFakeMemoryManager(FakeMemoryManager):
    def search_memory(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[MemoryRecord]:
        super().search_memory(query, memory_type, limit, session_id, **kwargs)
        return [
            MemoryRecord(
                id="ctx-mem-1",
                content="用户正在使用 Pandas 处理 CSV 数据",
                memory_type=memory_type,
                importance=0.8,
                metadata={"session_id": session_id or "", "created_at": 1_700_000_000.0},
            )
        ]


def _system_content(messages: LLMMessages) -> str:
    for message in messages:
        if message.get("role") == "system":
            return str(message.get("content", ""))
    return ""


def _history_role_messages(messages: LLMMessages) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") in {"user", "assistant"}]


def test_context_aware_agent_builds_six_section_system_message() -> None:
    manager = SearchableFakeMemoryManager()
    service = MemoryService(manager=manager, memory_types=["working"])
    memory_tool = MemoryTool(
        user_id="ctx_user",
        session_id="ctx_session",
        memory_types=["working"],
        memory_service=service,
    )
    llm = CapturingLLM()
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=RagTool(rag_manager=FakeRagManager()),
        config=ContextConfig(min_relevance=0.0),
    )

    agent = ContextAwareAgent(
        name="ctx-agent",
        llm=llm,
        system_prompt="你是数据工程顾问。",
        memory_service=service,
        memory_hooks=MemoryHookConfig(session_id="ctx_session"),
        context_builder=builder,
    )

    answer = agent.run("如何优化 Pandas 内存？")

    assert answer == "已结合上下文回答"
    system = _system_content(llm.last_messages)
    assert "[Role & Policies]" in system
    assert "[Task]" in system
    assert "[Evidence]" in system
    assert "[Context]" in system
    assert "## 相关记忆（自动检索）" not in system
    assert agent.last_built_context is not None
    assert agent.last_built_context.stats["selected_packets"] >= 1


def test_context_aware_agent_does_not_duplicate_history() -> None:
    from core.message import Message

    manager = SearchableFakeMemoryManager()
    service = MemoryService(manager=manager, memory_types=["working"])
    memory_tool = MemoryTool(
        user_id="ctx_user",
        session_id="ctx_session",
        memory_types=["working"],
        memory_service=service,
    )
    llm = CapturingLLM()
    builder = ContextBuilder(
        memory_tool=memory_tool,
        config=ContextConfig(min_relevance=0.0),
    )

    agent = ContextAwareAgent(
        name="ctx-agent",
        llm=llm,
        system_prompt="你是助手。",
        context_builder=builder,
    )
    agent.add_message(Message(content="第一轮用户问题", role="user"))
    agent.add_message(Message(content="第一轮助手回答", role="assistant"))

    agent.run("第二轮问题")

    role_messages = _history_role_messages(llm.last_messages)
    assert role_messages == [{"role": "user", "content": "第二轮问题"}]
    assert "第一轮用户问题" in _system_content(llm.last_messages)


def test_context_aware_agent_disables_legacy_retrieve_but_records() -> None:
    manager = FakeMemoryManager(memory_id="ctx_record_1")
    service = MemoryService(manager=manager, memory_types=["working"])
    memory_tool = MemoryTool(
        user_id="ctx_user",
        session_id="ctx_session",
        memory_types=["working"],
        memory_service=service,
    )
    llm = CapturingLLM()
    builder = ContextBuilder(memory_tool=memory_tool, config=ContextConfig(min_relevance=0.0))

    agent = ContextAwareAgent(
        name="ctx-agent",
        llm=llm,
        memory_service=service,
        memory_hooks=MemoryHookConfig(session_id="ctx_session"),
        context_builder=builder,
    )

    assert agent.memory_hooks is not None
    assert agent.memory_hooks.retrieve_before_run is False

    agent.run("记住我喜欢深色主题")

    assert "## 相关记忆（自动检索）" not in _system_content(llm.last_messages)
    assert len(manager.added) == 1
    assert "用户: 记住我喜欢深色主题" in manager.added[0]["content"]


def test_with_agent_tools_factory_wires_context_builder() -> None:
    manager = SearchableFakeMemoryManager()
    service = MemoryService(
        manager=manager,
        memory_types=["working", "episodic"],
    )
    llm = CapturingLLM()

    agent = ContextAwareAgent.with_agent_tools(
        name="ctx-factory",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_memory=True,
        enable_rag=False,
        memory_service=service,
        context_config=ContextConfig(min_relevance=0.0),
    )

    assert agent.context_builder is not None
    assert agent.memory_hooks is not None
    assert agent.memory_hooks.retrieve_before_run is False

    agent.run("查询记忆")

    assert "[Role & Policies]" in _system_content(llm.last_messages)
    assert manager.searches
