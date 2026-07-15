"""RAG reranker：对向量检索候选重排序。

两种实现：
- NoneReranker：透传，保持原有向量分数顺序（默认）
- LLMReranker：单次 LLM 调用批量打分，输出 JSON 分数数组后重排序
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from .models import RagSearchResult


class Reranker(Protocol):
    """Reranker 协议：接收候选结果列表，返回重排序后的列表。"""

    def rerank(
        self,
        query: str,
        results: list[RagSearchResult],
        top_k: int,
    ) -> list[RagSearchResult]:
        ...


class NoneReranker:
    """透传 reranker，保持向量检索原始顺序。"""

    def rerank(
        self,
        query: str,
        results: list[RagSearchResult],
        top_k: int,
    ) -> list[RagSearchResult]:
        return results[:top_k]


class LLMReranker:
    """LLM 打分 reranker。

    将所有候选片段拼入单次 LLM 调用，让模型输出 JSON 分数数组，
    按分数降序排列后截断 top_k。LLM 不可用或解析失败时回退到原始向量分数。
    """

    _JSON_ARRAY_PATTERN = re.compile(r"\[[^\]]*\]")

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def rerank(
        self,
        query: str,
        results: list[RagSearchResult],
        top_k: int,
    ) -> list[RagSearchResult]:
        if not results:
            return []

        scores = self._batch_score(query, results)
        scored = list(zip(scores, results, strict=True))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored[:top_k]]

    def _batch_score(
        self,
        query: str,
        results: list[RagSearchResult],
    ) -> list[float]:
        """单次 LLM 调用为所有候选打分；失败时回退到原始向量分数。"""
        passages = '\n\n'.join(
            f"[{i}] {result.chunk.content}"
            for i, result in enumerate(results, start=1)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是检索相关性评估器。根据用户问题对每个文本片段打相关性分数。"
                    "只输出一个 JSON 数组，数组长度等于片段数量，"
                    "每个元素为 0 到 1 之间的小数（1 表示完全相关，0 表示无关）。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{query}\n\n"
                    f"文本片段：\n{passages}\n\n"
                    f"相关性分数 JSON 数组："
                ),
            },
        ]
        try:
            raw = self._llm.invoke(messages, temperature=0) or ""
        except Exception:
            return [result.score for result in results]

        parsed = self._parse_scores(raw, expected=len(results))
        if parsed is None:
            return [result.score for result in results]
        return parsed

    def _parse_scores(self, raw: str, *, expected: int) -> list[float] | None:
        """从 LLM 输出解析分数数组，长度不匹配时返回 None。"""
        import json

        match = self._JSON_ARRAY_PATTERN.search(raw)
        if match is None:
            return None
        try:
            arr = json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(arr, list) or len(arr) != expected:
            return None
        try:
            return [max(0.0, min(1.0, float(v))) for v in arr]
        except (TypeError, ValueError):
            return None


def create_reranker(
    rerank: str | bool | None = None,
    llm: Any | None = None,
) -> Reranker:
    """根据参数创建 reranker。

    Args:
        rerank: "none"/False/None → NoneReranker；"llm"/True → LLMReranker
        llm: LLMReranker 所需的 LLM 实例
    """
    normalized = rerank
    if isinstance(normalized, bool):
        normalized = "llm" if normalized else "none"
    normalized = str(normalized or "none").strip().lower()

    if normalized in {"none", "off", "false", ""}:
        return NoneReranker()
    if normalized in {"llm", "on", "true"}:
        if llm is None:
            raise ValueError("LLM reranker 需要 llm 实例")
        return LLMReranker(llm)
    raise ValueError(f"不支持的 rerank 选项: {rerank}")
