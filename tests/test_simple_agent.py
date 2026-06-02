"""SimpleAgent smoke tests"""
from collections.abc import Iterator
from typing import Any

from agents.prompts import DEFAULT_SIMPLE_AGENT_SYSTEM_PROMPT
from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse, ToolCall
from tools.base import Tool
from tools.registry import ToolRegistry


class FakeLLM(BaseLLM):
    """不调用真实 API 的测试用 LLM"""

    def __init__(self) -> None:
        self.model = "fake-model"
        self.client = None  # type: ignore[assignment]
        self.last_messages: LLMMessages | None = None

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        self.last_messages = messages
        return "mock reply"

    def stream_invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield "mock"
        yield " reply"

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        return LLMToolResponse(content="should not use tools path", tool_calls=None)


class FakeToolLLM(FakeLLM):
    """模拟先调工具、再返回最终答案"""

    def __init__(self) -> None:
        super().__init__()
        self._round = 0

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        self._round += 1
        if self._round == 1:
            return LLMToolResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="add",
                        arguments='{"a": 1, "b": 2}',
                    )
                ],
            )
        return LLMToolResponse(content="结果是 3", tool_calls=None)

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        return "fallback"


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


def test_simple_agent_run() -> None:
    agent = SimpleAgent(name="test", llm=FakeLLM())
    result = agent.run("hello")
    assert result == "mock reply"
    history = agent.get_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "hello"
    assert history[1].role == "assistant"
    assert history[1].content == "mock reply"


def test_simple_agent_uses_default_system_prompt() -> None:
    llm = FakeLLM()
    agent = SimpleAgent(name="test", llm=llm)

    agent.run("hello")

    assert llm.last_messages is not None
    assert llm.last_messages[0] == {
        "role": "system",
        "content": DEFAULT_SIMPLE_AGENT_SYSTEM_PROMPT,
    }


def test_simple_agent_stream_run() -> None:
    agent = SimpleAgent(name="test", llm=FakeLLM())
    chunks = list(agent.stream_run("hi"))
    assert chunks == ["mock", " reply"]
    assert agent.get_history()[-1].content == "mock reply"


def test_simple_agent_tool_calling() -> None:
    registry = ToolRegistry()
    registry.register_tool(AddTool())
    agent = SimpleAgent(
        name="test",
        llm=FakeToolLLM(),
        tool_registry=registry,
        enable_tool_calling=True,
    )
    assert agent.has_tools()
    result = agent.run("1+2等于多少")
    assert result == "结果是 3"
    assert len(agent.list_tools()) == 1


def test_add_tool_lazy_registry() -> None:
    agent = SimpleAgent(name="test", llm=FakeToolLLM(), enable_tool_calling=False)
    assert not agent.has_tools()
    agent.add_tool(AddTool())
    assert agent.has_tools()
    result = agent.run("计算")
    assert result == "结果是 3"
