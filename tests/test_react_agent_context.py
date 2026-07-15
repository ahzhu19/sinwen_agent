"""ReActAgent ContextBuilder 集成测试。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agents.react_agent import ReActAgent
from context import ContextBuilder, ContextConfig
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse
from memory.hooks import MemoryHookConfig
from memory.modules.base import MemoryRecord
from memory.service import MemoryService
from tests.memory_fakes import FakeMemoryManager
from tests.test_rag_tool import FakeRagManager
from tools.base import Tool
from tools.builtin.memory_tool import MemoryTool
from tools.builtin.rag_tool import RagTool
from tools.registry import ToolRegistry


class ReActCapturingLLM(BaseLLM):
    """ReAct 测试 LLM，按序返回预设响应，记录每次调用的 messages。"""

    def __init__(self, responses: list[str]) -> None:
        self.model = "react-context"
        self.client = None  # type: ignore[assignment]
        self.responses = responses
        self.calls = 0
        self.all_messages: list[LLMMessages] = []

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        self.all_messages.append(messages)
        response = self.responses[self.calls]
        self.calls += 1
        return response

    def stream_invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> Iterator[str]:
        _ = messages, temperature, kwargs
        yield ""

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        _ = messages, tools, tool_choice, temperature, kwargs
        return LLMToolResponse(content=None, tool_calls=None)


class AddTool(Tool):
    @property
    def name(self) -> str:
        return "add"

    @property
    def description(self) -> str:
        return "将两个整数相加"

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "第一个数"},
                "b": {"type": "integer", "description": "第二个数"},
            },
            "required": ["a", "b"],
        }

    def run(self, **kwargs: Any) -> str:
        return str(kwargs["a"] + kwargs["b"])


class SearchableFakeMemoryManager(FakeMemoryManager):
    """search_memory 返回带 importance 的 MemoryRecord，供 ContextBuilder 评分。"""

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
                id="react-mem-1",
                content="用户偏好用 Python 处理数据分析",
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


def _user_content(messages: LLMMessages) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _build_agent(
    llm: ReActCapturingLLM,
    *,
    enable_context: bool = True,
    context_builder: ContextBuilder | None = None,
) -> ReActAgent:
    manager = SearchableFakeMemoryManager()
    service = MemoryService(manager=manager, memory_types=["working"])
    memory_tool = MemoryTool(
        user_id="react_user",
        session_id="react_session",
        memory_types=["working"],
        memory_service=service,
    )
    if context_builder is None:
        context_builder = ContextBuilder(
            memory_tool=memory_tool,
            rag_tool=RagTool(rag_manager=FakeRagManager()),
            config=ContextConfig(min_relevance=0.0),
        )
    registry = ToolRegistry()
    registry.register_tool(AddTool())
    return ReActAgent(
        name="react-ctx",
        llm=llm,
        tool_registry=registry,
        memory_service=service,
        memory_hooks=MemoryHookConfig(session_id="react_session"),
        context_builder=context_builder if enable_context else None,
        enable_context=enable_context,
    )


def test_react_context_builds_six_section_system_message() -> None:
    """启用 ContextBuilder 时，system 消息包含六分区骨架而非 flat memory 注入。"""
    llm = ReActCapturingLLM([
        "Thought: 已结合上下文。\nAction: Finish\nAction Input: Pandas 优化建议"
    ])
    agent = _build_agent(llm)

    result = agent.run("如何优化 Pandas 内存？")

    assert result == "Pandas 优化建议"
    system = _system_content(llm.all_messages[0])
    assert "[Role & Policies]" in system
    assert "[Task]" in system
    assert "[Evidence]" in system
    assert "[Context]" in system
    assert "[Output]" in system
    assert "## 相关记忆（自动检索）" not in system
    assert agent.last_built_context is not None


def test_react_context_react_trace_stays_in_user_prompt() -> None:
    """react_trace 保留在 user prompt（ReAct 模板），不进入 ContextBuilder 的 system 消息。"""
    llm = ReActCapturingLLM([
        'Thought: 需要先计算。\nAction: add\nAction Input: {"a": 1, "b": 2}',
        "Thought: 已得到结果。\nAction: Finish\nAction Input: 结果是 3",
    ])
    agent = _build_agent(llm)

    agent.run("1+2等于多少")

    assert llm.calls == 2
    step2_user = _user_content(llm.all_messages[1])
    assert "Thought: 需要先计算。" in step2_user
    assert "Observation: 3" in step2_user
    step2_system = _system_content(llm.all_messages[1])
    assert "Observation: 3" not in step2_system


def test_react_context_built_once_per_run() -> None:
    """多步 ReAct 循环中，ContextBuilder 只构建一次（记忆搜索只触发一次）。"""
    llm = ReActCapturingLLM([
        'Thought: 需要计算。\nAction: add\nAction Input: {"a": 1, "b": 2}',
        "Thought: 完成。\nAction: Finish\nAction Input: 结果是 3",
    ])
    agent = _build_agent(llm)

    agent.run("1+2等于多少")

    manager = agent.memory_service.manager  # type: ignore[attr-defined]
    working_searches = [s for s in manager.searches if s["memory_type"] == "working"]
    assert len(working_searches) == 1


def test_react_context_disables_legacy_retrieve_but_records() -> None:
    """启用 ContextBuilder 时关闭 legacy retrieve，但 record_after_run 仍保留。"""
    llm = ReActCapturingLLM([
        "Thought: 完成。\nAction: Finish\nAction Input: 深色主题已记住"
    ])
    agent = _build_agent(llm)

    assert agent.memory_hooks is not None
    assert agent.memory_hooks.retrieve_before_run is False

    agent.run("记住我喜欢深色主题")

    system = _system_content(llm.all_messages[0])
    assert "## 相关记忆（自动检索）" not in system
    manager = agent.memory_service.manager  # type: ignore[attr-defined]
    assert len(manager.added) == 1
    assert "用户: 记住我喜欢深色主题" in manager.added[0]["content"]


def test_react_context_legacy_path_when_disabled() -> None:
    """enable_context=False 时走旧路径（flat memory_context 注入），不构建六分区。"""
    llm = ReActCapturingLLM([
        "Thought: 完成。\nAction: Finish\nAction Input: 旧路径正常"
    ])
    agent = _build_agent(llm, enable_context=False)

    assert agent.context_builder is None
    agent.run("测试旧路径")

    system = _system_content(llm.all_messages[0])
    assert "[Role & Policies]" not in system
    assert agent.last_built_context is None


def test_react_with_agent_tools_wires_context_builder() -> None:
    """with_agent_tools 工厂方法装配 ContextBuilder 并关闭 legacy retrieve。"""
    manager = SearchableFakeMemoryManager()
    service = MemoryService(manager=manager, memory_types=["working", "episodic"])
    llm = ReActCapturingLLM([
        "Thought: 完成。\nAction: Finish\nAction Input: 工厂装配正常"
    ])

    agent = ReActAgent.with_agent_tools(
        name="react-factory",
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

    system = _system_content(llm.all_messages[0])
    assert "[Role & Policies]" in system
    assert "[Evidence]" in system
    assert manager.searches
