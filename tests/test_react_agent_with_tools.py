"""ReActAgent.with_agent_tools integration tests."""

from collections.abc import Iterator
from typing import Any

from agents.react_agent import ReActAgent
from core.llm import BaseLLM, LLMMessages
from tests.memory_fakes import FakeMemoryManager
from tools.agent_registry import create_agent_tool_registry
from tools.builtin.memory_tool import MemoryTool


class FakeReActMemoryLLM(BaseLLM):
    def __init__(self) -> None:
        self.model = "fake-react-memory"
        self.client = None  # type: ignore[assignment]
        self.calls = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = messages, temperature, kwargs
        self.calls += 1
        if self.calls == 1:
            return (
                'Thought: 先查统计。\n'
                'Action: memory\n'
                'Action Input: {"action": "stats"}'
            )
        return 'Thought: 完成。\nAction: Finish\nAction Input: 当前有 3 条工作记忆'

    def stream_invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> Iterator[str]:
        _ = messages, temperature, kwargs
        yield ""


def test_registry_react_can_enable_memory() -> None:
    registry = create_agent_tool_registry(
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
    )
    assert "memory" in registry.list_tools()


def test_react_agent_with_agent_tools_runs_memory() -> None:
    llm = FakeReActMemoryLLM()
    agent = ReActAgent.with_agent_tools(
        name="react-mem",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        memory_types=["working"],
    )
    agent.tool_registry.unregister_tool("memory")
    agent.tool_registry.register_tool(
        MemoryTool(memory_manager=FakeMemoryManager(stats_counts={"working": 3}))
    )

    result = agent.run("当前有多少条记忆？")

    assert result == "当前有 3 条工作记忆"
    assert llm.calls == 2
    trace = "\n".join(agent.react_trace)
    assert "working: 3 条" in trace
