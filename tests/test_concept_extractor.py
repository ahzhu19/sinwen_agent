"""概念抽取器测试。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memory.concept_extractor import LLMConceptExtractor
from memory.config import MemoryConfig


def test_llm_prefers_metadata_concepts() -> None:
    config = MemoryConfig(
        llm_api_key="test-key",
        llm_base_url="http://127.0.0.1:9",
        llm_model_id="test-model",
    )
    extractor = LLMConceptExtractor(config)
    concepts = extractor.extract("ignored", {"concepts": ["Neo4j", "Milvus"]})
    assert concepts == ["Neo4j", "Milvus"]


def test_llm_extracts_via_chat_api() -> None:
    config = MemoryConfig(
        llm_api_key="test-key",
        llm_base_url="http://127.0.0.1:9",
        llm_model_id="test-model",
    )
    extractor = LLMConceptExtractor(config)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"concepts": ["PostgreSQL", "Neo4j"]}'

    with patch.object(extractor._client.chat.completions, "create", return_value=mock_response):
        result = extractor.extract_with_source("用户偏好 PostgreSQL 与 Neo4j", {})

    assert result.source == "llm"
    assert result.concepts == ["PostgreSQL", "Neo4j"]


def test_llm_raises_on_api_error() -> None:
    config = MemoryConfig(
        llm_api_key="test-key",
        llm_base_url="http://127.0.0.1:9",
        llm_model_id="test-model",
    )
    extractor = LLMConceptExtractor(config)

    with patch.object(
        extractor._client.chat.completions,
        "create",
        side_effect=ConnectionError("down"),
    ):
        with pytest.raises(RuntimeError, match="LLM 概念抽取失败"):
            extractor.extract("语义记忆结合图数据库", {})
