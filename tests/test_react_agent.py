"""ReActAgent tests."""
from collections.abc import Iterator
from typing import Any

import pytest

from prompts import render_prompt
from agents.react_agent import ReActAgent, parse_react_output
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse
from tools.base import Tool
from tools.registry import ToolRegistry


class FakeReActLLM(BaseLLM):
    """不调用真实 API 的 ReAct 测试 LLM。"""

    def __init__(self, responses: list[str | None]) -> None:
        self.model = "fake-react-model"
        self.client = None  # type: ignore[assignment]
        self.responses = responses
        self.calls = 0
        self.last_messages: LLMMessages | None = None

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        self.last_messages = messages
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


def test_parse_react_output_with_json_action_input() -> None:
    step = parse_react_output(
        'Thought: 需要计算。\nAction: add\nAction Input: {"a": 1, "b": 2}'
    )

    assert step.thought == "需要计算。"
    assert step.action_name == "add"
    assert step.action_input == {"a": 1, "b": 2}
    assert not step.is_finish


def test_react_agent_runs_action_then_finish() -> None:
    registry = ToolRegistry()
    registry.register_tool(AddTool())
    llm = FakeReActLLM([
        'Thought: 需要先计算。\nAction: add\nAction Input: {"a": 1, "b": 2}',
        "Thought: 已经得到结果。\nAction: Finish\nAction Input: 结果是 3",
    ])
    agent = ReActAgent(name="react", llm=llm, tool_registry=registry)

    result = agent.run("1+2等于多少")

    assert result == "结果是 3"
    assert llm.calls == 2
    assert "Observation: 3" in "\n".join(agent.react_trace)
    history = agent.get_history()
    assert history[0].content == "1+2等于多少"
    assert history[1].content == "结果是 3"


def test_react_agent_returns_fallback_after_max_steps() -> None:
    registry = ToolRegistry()
    registry.register_tool(AddTool())
    llm = FakeReActLLM([
        'Thought: 继续计算。\nAction: add\nAction Input: {"a": 1, "b": 2}',
    ])
    agent = ReActAgent(name="react", llm=llm, tool_registry=registry, max_steps=1)

    result = agent.run("1+2等于多少")

    assert result == "抱歉，我无法在限定步数内完成这个任务。"
    assert len(agent.get_history()) == 2


def test_react_agent_accepts_custom_user_prompt_template() -> None:
    registry = ToolRegistry()
    registry.register_tool(AddTool())
    llm = FakeReActLLM([
        "Thought: 已经可以回答。\nAction: Finish\nAction Input: 自定义模板正常"
    ])
    agent = ReActAgent(
        name="react",
        llm=llm,
        tool_registry=registry,
        user_prompt_template="TOOLS={tools}\nQUESTION={question}\nTRACE={history}",
    )

    result = agent.run("测试模板")

    assert result == "自定义模板正常"
    assert llm.last_messages is not None
    assert llm.last_messages[-1]["content"].startswith("TOOLS=- add")
    assert "QUESTION=测试模板" in llm.last_messages[-1]["content"]


def test_render_prompt_reports_missing_variable() -> None:
    with pytest.raises(ValueError, match="Prompt 缺少变量"):
        render_prompt("{missing}", question="hello")
