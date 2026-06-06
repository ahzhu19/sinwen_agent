"""语义记忆概念抽取：metadata 优先 + LLM。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI

from prompts.memory import build_concept_extraction_messages

from .config import MemoryConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptExtractionResult:
    concepts: list[str]
    source: str  # metadata | llm
    error: str | None = None


class ConceptExtractor(Protocol):
    def extract(self, content: str, metadata: dict[str, Any]) -> list[str]:
        ...


class LLMConceptExtractor:
    """OpenAI-compatible Chat 概念抽取。"""

    def __init__(self, config: MemoryConfig) -> None:
        if not config.llm_api_key or not config.llm_base_url:
            raise ValueError(
                "语义记忆概念抽取需要 LLM_API_KEY 与 LLM_BASE_URL（未设置时回退 EMBED_BASE_URL）"
            )
        self._config = config
        self._model = config.llm_model_id
        self._max_concepts = config.concept_extraction_max_concepts
        self._client = OpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            timeout=60,
        )

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
                    concepts=_dedupe(concepts)[: self._max_concepts],
                    source="metadata",
                )

        messages = build_concept_extraction_messages(
            content,
            max_concepts=self._max_concepts,
            metadata=metadata,
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
            )
            text = (response.choices[0].message.content or "").strip()
            concepts = _parse_concepts_json(text)
            if concepts:
                return ConceptExtractionResult(
                    concepts=_dedupe(concepts)[: self._max_concepts],
                    source="llm",
                )
            raise RuntimeError("LLM 返回空概念列表")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"LLM 概念抽取失败: {exc}") from exc


def extract_concepts(
    extractor: ConceptExtractor,
    content: str,
    metadata: dict[str, Any],
) -> ConceptExtractionResult:
    if hasattr(extractor, "extract_with_source"):
        return extractor.extract_with_source(content, metadata)  # type: ignore[attr-defined]
    return ConceptExtractionResult(
        concepts=extractor.extract(content, metadata),
        source="llm",
    )


def create_concept_extractor(config: MemoryConfig) -> LLMConceptExtractor:
    return LLMConceptExtractor(config)


def _parse_concepts_json(text: str) -> list[str]:
    payload = text
    if "```" in payload:
        payload = payload.split("```", 2)[1]
        if payload.startswith("json"):
            payload = payload[4:]
    data = json.loads(payload.strip())
    raw = data.get("concepts", [])
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
