"""PlanAndSolveAgent tests."""
from collections.abc import Iterator
from typing import Any

from agents.plan_and_solve_agent import PlanAndSolveAgent, parse_plan
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse


class ScriptedLLM(BaseLLM):
    """按预设顺序返回文本的测试 LLM。"""

    def __init__(self, responses: list[str | None]) -> None:
        self.model = "fake-plan-solve-model"
        self.client = None  # type: ignore[assignment]
        self.responses = responses
        self.calls = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        response = self.responses[self.calls]
        self.calls += 1
        return response

    def stream_invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield ""

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        return LLMToolResponse(content=None, tool_calls=None)


def test_parse_plan_pure_list() -> None:
    assert parse_plan('["第一步", "第二步"]') == ["第一步", "第二步"]


def test_parse_plan_with_surrounding_text() -> None:
    text = '好的，计划如下：\n["分析题目", "给出答案"]\n请执行。'
    assert parse_plan(text) == ["分析题目", "给出答案"]


def test_parse_plan_invalid_returns_empty() -> None:
    assert parse_plan("这不是列表") == []
    assert parse_plan("[]") == []
    assert parse_plan("") == []


def test_plan_and_solve_happy_path() -> None:
    llm = ScriptedLLM([
        '["第一步：分析", "第二步：计算"]',
        "步骤1结果",
        "步骤2结果",
        "最终汇总答案",
    ])
    agent = PlanAndSolveAgent(name="plan-solve", llm=llm)

    result = agent.run("1+2等于多少")

    assert result == "最终汇总答案"
    assert llm.calls == 4
    assert "Plan:" in agent.plan_trace[0]
    history = agent.get_history()
    assert history[0].content == "1+2等于多少"
    assert history[1].content == "最终汇总答案"


def test_plan_retry_then_success() -> None:
    llm = ScriptedLLM([
        "抱歉，我无法输出列表",
        '["唯一步骤"]',
        "步骤结果",
        "最终答案",
    ])
    agent = PlanAndSolveAgent(name="plan-solve", llm=llm, max_plan_retries=2)

    result = agent.run("简单问题")

    assert result == "最终答案"
    assert llm.calls == 4


def test_plan_parse_fails_returns_friendly_message() -> None:
    llm = ScriptedLLM([
        "脏数据",
        "还是脏数据",
        "仍然脏数据",
        "仍然脏数据",
    ])
    agent = PlanAndSolveAgent(name="plan-solve", llm=llm, max_plan_retries=3)

    result = agent.run("无法规划的问题")

    assert result == "抱歉，我无法为这个问题制定有效的计划。"
    assert llm.calls == 4
    assert len(agent.get_history()) == 2


def test_step_failure_continues_and_synthesis_mentions_incomplete() -> None:
    llm = ScriptedLLM([
        '["步骤A", "步骤B"]',
        None,
        "步骤B成功",
        "最终答案（含未完成说明）",
    ])
    agent = PlanAndSolveAgent(name="plan-solve", llm=llm)

    result = agent.run("两步骤任务")

    assert result == "最终答案（含未完成说明）"
    assert llm.calls == 4
    assert any("[此步未能完成]" in entry for entry in agent.plan_trace)
