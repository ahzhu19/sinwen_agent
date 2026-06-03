"""RAG answer generation."""

from __future__ import annotations

from typing import Any

from .models import RagAnswer, RagSearchResult


class RagGenerator:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def answer(self, query: str, sources: list[RagSearchResult]) -> RagAnswer:
        if not sources:
            return RagAnswer(answer="无法从知识库中找到足够信息回答该问题。", sources=[])
        context = self._assemble_context(sources)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个严格基于知识库上下文回答的助手。"
                    "只使用提供的上下文回答；如果上下文不足，明确说明无法确认。"
                    "回答中必须引用来源编号，例如 [Source 1]。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{query}\n\n上下文：\n{context}",
            },
        ]
        response = self._llm.invoke(messages, temperature=0)
        if not response:
            return RagAnswer(answer="生成回答失败。", sources=sources)
        return RagAnswer(answer=response, sources=sources)

    def _assemble_context(self, sources: list[RagSearchResult]) -> str:
        parts: list[str] = []
        for index, result in enumerate(sources, start=1):
            heading = " / ".join(result.chunk.heading_path) or "(无标题)"
            parts.append(
                "\n".join(
                    [
                        f"[Source {index}]",
                        f"Document: {result.document.title or result.document.source_uri}",
                        f"Heading: {heading}",
                        "Content:",
                        result.chunk.content,
                    ]
                )
            )
        return "\n\n".join(parts)
