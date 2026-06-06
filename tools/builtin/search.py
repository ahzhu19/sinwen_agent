"""智能混合搜索工具。

支持多种搜索引擎后端，智能选择最佳搜索源：

1. 混合模式 (hybrid) - 按查询类型选主后端，失败再故障转移
2. Tavily API (tavily) - 专业 AI 搜索
3. SerpApi (serpapi) - 传统 Google 搜索
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from tools.base import Tool

if TYPE_CHECKING:
    from core.llm import BaseLLM

_TIME_SENSITIVE_KEYWORDS = (
    "最新",
    "新闻",
    "今天",
    "今日",
    "实时",
    "近期",
    "最近",
    "latest",
    "news",
    "today",
    "breaking",
)
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass
class SearchResult:
    """单条搜索结果。"""

    title: str
    url: str
    snippet: str


class SearchBackend(ABC):
    """搜索后端接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称。"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """后端是否可用（通常取决于密钥是否配置）。"""

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """执行搜索，返回结果列表。"""


class _TavilyBackend(SearchBackend):
    """基于 Tavily 官方 SDK 的后端。"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=self._api_key)
        response = client.search(query=query, max_results=max_results)

        results: list[SearchResult] = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                )
            )
        return results


class _SerpApiBackend(SearchBackend):
    """基于 SerpApi 官方 SDK 的后端。"""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.getenv("SERPAPI_API_KEY")

    @property
    def name(self) -> str:
        return "serpapi"

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        from serpapi import GoogleSearch

        search = GoogleSearch({
            "q": query,
            "api_key": self._api_key,
            "num": max_results,
        })
        data = search.get_dict()

        results: list[SearchResult] = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                )
            )
        return results


def _is_time_sensitive(query: str) -> bool:
    lowered = query.lower()
    if any(keyword in lowered for keyword in _TIME_SENSITIVE_KEYWORDS):
        return True
    return bool(_YEAR_PATTERN.search(query))


class SearchTool(Tool):
    """智能混合搜索工具。"""

    def __init__(
        self,
        mode: str = "hybrid",
        max_results: int = 5,
        routing: str = "keyword",
        llm: Optional["BaseLLM"] = None,
        tavily_backend: Optional[SearchBackend] = None,
        serpapi_backend: Optional[SearchBackend] = None,
    ) -> None:
        self.mode = mode
        self.max_results = max_results
        self.routing = routing
        self._llm = llm
        self._tavily = tavily_backend or _TavilyBackend()
        self._serpapi = serpapi_backend or _SerpApiBackend()

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "联网搜索网页信息，返回相关结果的标题、链接和摘要"

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题",
                },
            },
            "required": ["query"],
        }

    def run(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "错误：query 不能为空"

        max_results = int(kwargs.get("max_results") or self.max_results)
        backends = self._select_backends(query)

        available_backends = [b for b in backends if b.available]
        if not available_backends:
            return (
                "错误：未配置任何可用的搜索后端密钥"
                "（TAVILY_API_KEY / SERPAPI_API_KEY）"
            )

        for backend in available_backends:
            try:
                results = backend.search(query, max_results)
            except Exception as e:
                print(f"⚠️ 搜索后端 {backend.name} 调用失败: {e}")
                continue
            if results:
                return self._format_results(results)

        return "未找到相关结果"

    def _select_backends(self, query: str) -> list[SearchBackend]:
        if self.mode == "tavily":
            return [self._tavily]
        if self.mode == "serpapi":
            return [self._serpapi]

        if self._route_time_sensitive(query):
            return [self._tavily, self._serpapi]
        return [self._serpapi, self._tavily]

    def _route_time_sensitive(self, query: str) -> bool:
        if self.routing == "llm" and self._llm is not None:
            return self._llm_route_time_sensitive(query)
        return _is_time_sensitive(query)

    def _llm_route_time_sensitive(self, query: str) -> bool:
        from prompts import SEARCH_ROUTING_PROMPT, render_prompt

        prompt = render_prompt(SEARCH_ROUTING_PROMPT, query=query)
        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])  # type: ignore[union-attr]
        except Exception as e:
            print(f"⚠️ LLM 路由失败，回退到关键词法: {e}")
            return _is_time_sensitive(query)
        if not response:
            return _is_time_sensitive(query)
        return "TAVILY" in response.upper()

    @staticmethod
    def _format_results(results: list[SearchResult]) -> str:
        blocks: list[str] = []
        for index, result in enumerate(results, 1):
            blocks.append(
                f"{index}. {result.title}\n   {result.url}\n   {result.snippet}"
            )
        return "\n\n".join(blocks)
