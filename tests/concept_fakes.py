"""概念抽取测试桩（不调用 LLM）。"""

from __future__ import annotations

from typing import Any

from memory.concept_extractor import ConceptExtractionResult


class StubConceptExtractor:
    """单元测试用：metadata.concepts 优先；否则返回预设 llm/query 概念。"""

    def __init__(
        self,
        *,
        max_concepts: int = 8,
        llm_concepts: list[str] | None = None,
        query_concepts: list[str] | None = None,
    ) -> None:
        self._max_concepts = max_concepts
        self._llm_concepts = list(llm_concepts or [])
        self._query_concepts = query_concepts

    def extract(self, content: str, metadata: dict[str, Any]) -> list[str]:
        return self.extract_with_source(content, metadata).concepts

    def extract_with_source(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> ConceptExtractionResult:
        raw_concepts = metadata.get("concepts")
        if isinstance(raw_concepts, list):
            concepts = [str(item).strip() for item in raw_concepts if str(item).strip()]
            if concepts:
                return ConceptExtractionResult(
                    concepts=concepts[: self._max_concepts],
                    source="metadata",
                )

        if not metadata and self._query_concepts is not None:
            return ConceptExtractionResult(
                concepts=self._query_concepts[: self._max_concepts],
                source="llm",
            )

        if self._llm_concepts:
            return ConceptExtractionResult(
                concepts=self._llm_concepts[: self._max_concepts],
                source="llm",
            )

        return ConceptExtractionResult(concepts=[], source="llm")
