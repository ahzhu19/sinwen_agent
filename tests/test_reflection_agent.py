"""ReflectionAgent tests."""
from collections.abc import Iterator
from typing import Any

from agents.reflection_agent import ReflectionAgent
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse


class ScriptedLLM(BaseLLM):
    """按预设脚本顺序返回文本的测试 LLM。"""

    def __init__(self, responses: list[str | None]) -> None:
        self.model = "fake-reflection-model"
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


def test_reflection_stops_early_when_critique_is_satisfied() -> None:
    llm = ScriptedLLM([
        "初稿答案",
        "NO_CHANGES",
    ])
    agent = ReflectionAgent(name="reflect", llm=llm, max_iterations=3)

    result = agent.run("写一句问候")

    assert result == "初稿答案"
    assert llm.calls == 2
    history = agent.get_history()
    assert history[0].content == "写一句问候"
    assert history[1].content == "初稿答案"


def test_reflection_revises_then_accepts() -> None:
    llm = ScriptedLLM([
        "初稿答案",
        "建议补充更多细节",
        "改进后的答案",
        "NO_CHANGES",
    ])
    agent = ReflectionAgent(name="reflect", llm=llm, max_iterations=3)

    result = agent.run("写一段介绍")

    assert result == "改进后的答案"
    assert llm.calls == 4


def test_reflection_returns_last_answer_at_max_iterations() -> None:
    llm = ScriptedLLM([
        "初稿答案",
        "还不够好",
        "第二版答案",
        "依然不够好",
        "第三版答案",
    ])
    agent = ReflectionAgent(name="reflect", llm=llm, max_iterations=2)

    result = agent.run("写点东西")

    assert result == "第三版答案"
    assert llm.calls == 5
    assert len(agent.get_history()) == 2
