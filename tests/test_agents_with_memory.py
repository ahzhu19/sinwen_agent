"""ReflectionAgent / PlanAndSolveAgent + MemoryTool 集成测试。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from agents.plan_and_solve_agent import PlanAndSolveAgent
from agents.reflection_agent import ReflectionAgent
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse, ToolCall
from tests.memory_fakes import FakeMemoryManager
from tools.builtin.memory_tool import MemoryTool


class FakeReflectionMemoryLLM(BaseLLM):
    def __init__(self) -> None:
        self.model = "fake-reflection-memory"
        self.client = None  # type: ignore[assignment]
        self.tool_round = 0
        self.invoke_calls = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = messages, temperature, kwargs
        self.invoke_calls += 1
        return "NO_CHANGES"

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
        self.tool_round += 1
        if self.tool_round == 1:
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
        return LLMToolResponse(content="结合记忆统计的初稿", tool_calls=None)


class HybridPlanMemoryLLM(BaseLLM):
    def __init__(self) -> None:
        self.model = "fake-plan-memory"
        self.client = None  # type: ignore[assignment]
        self.invoke_responses = ['["查工作记忆统计"]', "汇总后的最终答案"]
        self.invoke_idx = 0
        self.tool_round = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = messages, temperature, kwargs
        response = self.invoke_responses[self.invoke_idx]
        self.invoke_idx += 1
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
        self.tool_round += 1
        if self.tool_round == 1:
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
        return LLMToolResponse(content="步骤结果：working 2 条", tool_calls=None)


def _inject_fake_memory(agent: ReflectionAgent | PlanAndSolveAgent) -> FakeMemoryManager:
    assert agent.tool_registry is not None
    fake = FakeMemoryManager()
    agent.tool_registry.unregister_tool("memory")
    agent.tool_registry.register_tool(MemoryTool(memory_manager=fake))
    return fake


def test_reflection_with_agent_tools_registers_memory() -> None:
    agent = ReflectionAgent.with_agent_tools(
        name="reflect-mem",
        llm=FakeReflectionMemoryLLM(),
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
    )
    assert agent.tool_registry is not None
    assert "memory" in agent.tool_registry.list_tools()


def test_reflection_initial_phase_calls_memory_tool() -> None:
    llm = FakeReflectionMemoryLLM()
    agent = ReflectionAgent.with_agent_tools(
        name="reflect-mem",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
    )
    fake = _inject_fake_memory(agent)

    result = agent.run("写一段介绍")

    assert result == "结合记忆统计的初稿"
    assert llm.tool_round >= 1
    assert fake.stats_counts["working"] == 2


def test_plan_and_solve_with_agent_tools_registers_memory() -> None:
    agent = PlanAndSolveAgent.with_agent_tools(
        name="plan-mem",
        llm=HybridPlanMemoryLLM(),
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
    )
    assert agent.tool_registry is not None
    assert "memory" in agent.tool_registry.list_tools()


def test_plan_and_solve_solve_step_calls_memory_tool() -> None:
    llm = HybridPlanMemoryLLM()
    agent = PlanAndSolveAgent.with_agent_tools(
        name="plan-mem",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
    )
    _inject_fake_memory(agent)

    result = agent.run("统计记忆")

    assert result == "汇总后的最终答案"
    assert llm.tool_round >= 1
