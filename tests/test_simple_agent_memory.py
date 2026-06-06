"""SimpleAgent + MemoryTool integration tests."""

from typing import Any

from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse, ToolCall
from tests.memory_fakes import FakeMemoryManager
from tools.agent_registry import create_agent_tool_registry
from tools.builtin.memory_tool import MemoryTool


class FakeMemoryLLM(BaseLLM):
    def __init__(self) -> None:
        self.model = "fake-memory"
        self.client = None  # type: ignore[assignment]
        self._round = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = messages, temperature, kwargs
        return "已根据记忆统计回答"

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        _ = messages, tools, tool_choice, temperature, kwargs
        self._round += 1
        if self._round == 1:
            return LLMToolResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_mem",
                        name="memory",
                        arguments='{"action": "stats"}',
                    )
                ],
            )
        return LLMToolResponse(content="已根据记忆统计回答", tool_calls=None)


def test_registry_can_enable_memory() -> None:
    registry = create_agent_tool_registry(
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        memory_types=["working"],
    )
    assert "memory" in registry.list_tools()


def test_simple_agent_executes_memory_tool() -> None:
    registry = create_agent_tool_registry(
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
    )
    registry.unregister_tool("memory")
    registry.register_tool(MemoryTool(memory_manager=FakeMemoryManager()))

    agent = SimpleAgent(
        name="记忆助手",
        llm=FakeMemoryLLM(),
        tool_registry=registry,
        enable_tool_calling=True,
    )
    answer = agent.run("帮我看看当前有多少条记忆")
    assert "working: 2" in answer or "已根据记忆统计" in answer
