"""SearchTool tests."""
from collections.abc import Iterator
from typing import Any

from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse
from tools.builtin.search import SearchBackend, SearchResult, SearchTool


class FakeRoutingLLM(BaseLLM):
    """返回固定路由判定的假 LLM。"""

    def __init__(self, response: str | None) -> None:
        self.model = "fake-routing-model"
        self.client = None  # type: ignore[assignment]
        self._response = response
        self.calls = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        self.calls += 1
        return self._response

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


class FakeBackend(SearchBackend):
    """注入用的假搜索后端。"""

    def __init__(
        self,
        name: str,
        available: bool = True,
        results: list[SearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self._available = available
        self._results = results or []
        self._error = error
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def available(self) -> bool:
        return self._available

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._results


def _result(title: str) -> SearchResult:
    return SearchResult(title=title, url=f"https://example.com/{title}", snippet=f"{title} 摘要")


def test_hybrid_prefers_tavily_for_time_sensitive_query() -> None:
    tavily = FakeBackend("tavily", results=[_result("实时结果")])
    serpapi = FakeBackend("serpapi", results=[_result("通用结果")])
    tool = SearchTool(mode="hybrid", tavily_backend=tavily, serpapi_backend=serpapi)

    output = tool.run(query="今天的最新新闻")

    assert "实时结果" in output
    assert tavily.calls == 1
    assert serpapi.calls == 0


def test_hybrid_prefers_serpapi_for_general_query() -> None:
    tavily = FakeBackend("tavily", results=[_result("实时结果")])
    serpapi = FakeBackend("serpapi", results=[_result("通用结果")])
    tool = SearchTool(mode="hybrid", tavily_backend=tavily, serpapi_backend=serpapi)

    output = tool.run(query="python 列表用法")

    assert "通用结果" in output
    assert serpapi.calls == 1
    assert tavily.calls == 0


def test_hybrid_falls_back_when_primary_fails() -> None:
    tavily = FakeBackend("tavily", error=RuntimeError("api down"))
    serpapi = FakeBackend("serpapi", results=[_result("备用结果")])
    tool = SearchTool(mode="hybrid", tavily_backend=tavily, serpapi_backend=serpapi)

    output = tool.run(query="今天的最新新闻")

    assert "备用结果" in output
    assert tavily.calls == 1
    assert serpapi.calls == 1


def test_returns_error_when_no_backend_available() -> None:
    tavily = FakeBackend("tavily", available=False)
    serpapi = FakeBackend("serpapi", available=False)
    tool = SearchTool(mode="hybrid", tavily_backend=tavily, serpapi_backend=serpapi)

    output = tool.run(query="任意问题")

    assert output.startswith("错误：")


def test_formats_results_with_title_url_snippet() -> None:
    serpapi = FakeBackend("serpapi", results=[_result("结果A")])
    tavily = FakeBackend("tavily", available=False)
    tool = SearchTool(mode="hybrid", tavily_backend=tavily, serpapi_backend=serpapi)

    output = tool.run(query="python 教程")

    assert "结果A" in output
    assert "https://example.com/结果A" in output
    assert "结果A 摘要" in output


def test_single_serpapi_mode_ignores_time_sensitive_routing() -> None:
    tavily = FakeBackend("tavily", results=[_result("实时结果")])
    serpapi = FakeBackend("serpapi", results=[_result("通用结果")])
    tool = SearchTool(mode="serpapi", tavily_backend=tavily, serpapi_backend=serpapi)

    output = tool.run(query="今天的最新新闻")

    assert "通用结果" in output
    assert serpapi.calls == 1
    assert tavily.calls == 0


def test_llm_routing_overrides_keyword_decision() -> None:
    tavily = FakeBackend("tavily", results=[_result("实时结果")])
    serpapi = FakeBackend("serpapi", results=[_result("通用结果")])
    llm = FakeRoutingLLM("TAVILY")
    tool = SearchTool(
        mode="hybrid",
        routing="llm",
        llm=llm,
        tavily_backend=tavily,
        serpapi_backend=serpapi,
    )

    # 关键词法会判定为通用查询（无时效词），但 LLM 强制选 Tavily
    output = tool.run(query="python 列表用法")

    assert "实时结果" in output
    assert llm.calls == 1
    assert tavily.calls == 1
    assert serpapi.calls == 0


def test_llm_routing_falls_back_to_keyword_without_llm() -> None:
    tavily = FakeBackend("tavily", results=[_result("实时结果")])
    serpapi = FakeBackend("serpapi", results=[_result("通用结果")])
    tool = SearchTool(
        mode="hybrid",
        routing="llm",
        tavily_backend=tavily,
        serpapi_backend=serpapi,
    )

    # 未提供 llm，应回退到关键词法：通用查询走 SerpApi
    output = tool.run(query="python 列表用法")

    assert "通用结果" in output
    assert serpapi.calls == 1
    assert tavily.calls == 0
