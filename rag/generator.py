"""RAG answer generation."""

from __future__ import annotations

from typing import Any

from prompts.rag import RAG_ANSWER_SYSTEM_PROMPT, RAG_ANSWER_USER_PROMPT_TEMPLATE
from prompts.render import render_prompt

from .models import RagAnswer, RagSearchResult


class RagGenerator:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def answer(self, query: str, sources: list[RagSearchResult]) -> RagAnswer:
        if not sources:
            return RagAnswer(answer="无法从知识库中找到足够信息回答该问题。", sources=[])
        context = self._assemble_context(sources)
        messages = [
            {"role": "system", "content": RAG_ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_prompt(
                    RAG_ANSWER_USER_PROMPT_TEMPLATE,
                    query=query,
                    context=context,
                ),
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
