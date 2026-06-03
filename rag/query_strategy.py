"""RAG 查询策略。"""

from __future__ import annotations

from typing import Any, Protocol


class QueryStrategy(Protocol):
    def build_queries(self, query: str) -> list[str]:
        ...


class DirectQueryStrategy:
    def build_queries(self, query: str) -> list[str]:
        return [query.strip()] if query.strip() else []


class HyDEQueryStrategy:
    """用假设文档增强检索查询。"""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def build_queries(self, query: str) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "根据用户问题写一段简短、信息密集的假设性知识库段落，"
                    "用于向量检索。只输出段落正文，不要解释。"
                ),
            },
            {"role": "user", "content": query},
        ]
        hypothetical = self._llm.invoke(messages, temperature=0) or ""
        queries = [query]
        if hypothetical.strip():
            queries.append(hypothetical.strip())
        return queries


class MultiQueryStrategy:
    """生成多个子查询并合并检索。"""

    def __init__(self, llm: Any, subquery_count: int = 3) -> None:
        self._llm = llm
        self._subquery_count = subquery_count

    def build_queries(self, query: str) -> list[str]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"将用户问题改写为 {self._subquery_count} 个不同角度的检索子问题。"
                    "只输出 Python 列表字面量，例如 [\"子问题1\", \"子问题2\"]。"
                ),
            },
            {"role": "user", "content": query},
        ]
        raw = self._llm.invoke(messages, temperature=0) or "[]"
        subqueries = _parse_string_list(raw)
        if not subqueries:
            return [query]
        seen = {query}
        merged = [query]
        for item in subqueries:
            if item not in seen:
                seen.add(item)
                merged.append(item)
        return merged


def create_query_strategy(strategy: str, llm: Any | None = None) -> QueryStrategy:
    normalized = (strategy or "direct").strip().lower()
    if normalized in {"direct", "default", ""}:
        return DirectQueryStrategy()
    if normalized == "hyde":
        if llm is None:
            raise ValueError("HyDE 策略需要 LLM 实例")
        return HyDEQueryStrategy(llm)
    if normalized in {"multi", "multi_query", "mqe"}:
        if llm is None:
            raise ValueError("multi_query 策略需要 LLM 实例")
        return MultiQueryStrategy(llm)
    raise ValueError(f"不支持的查询策略: {strategy}")


def _parse_string_list(raw: str) -> list[str]:
    import ast

    text = raw.strip()
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
