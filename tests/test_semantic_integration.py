"""可选的真实 Neo4j + Milvus 语义记忆集成测试。

运行方式：

    RUN_SEMANTIC_INTEGRATION=1 uv run pytest tests/test_semantic_integration.py -v

需本地 Neo4j、Milvus、EMBED_API_KEY；LLM 概念抽取需 LLM_API_KEY（及 LLM_BASE_URL 或 EMBED_BASE_URL）。
有 DATABASE_URL 时建议另开 worker：``uv run python scripts/memory_vector_worker.py --once``
"""

from __future__ import annotations

import os

import pytest

from memory.config import MemoryConfig
from memory.manager import MemoryManager


@pytest.mark.skipif(
    os.getenv("RUN_SEMANTIC_INTEGRATION") != "1",
    reason="需要本地 Neo4j、Milvus 与 EMBED_API_KEY",
)
def test_semantic_memory_real_neo4j_milvus_roundtrip() -> None:
    config = MemoryConfig.from_env()
    if not config.embed_api_key:
        pytest.skip("未配置 EMBED_API_KEY")
    if not config.neo4j_uri:
        pytest.skip("未配置 NEO4J_URI")

    manager = MemoryManager(
        config=config,
        user_id="semantic_integration_user",
        enable_working=False,
        enable_episodic=False,
        enable_semantic=True,
        enable_perceptual=False,
    )

    memory_id = manager.add_memory(
        content="集成测试：用户偏好使用 PostgreSQL 与 Neo4j 构建记忆系统",
        memory_type="semantic",
        importance=0.85,
        metadata={
            "session_id": "semantic_integration_session",
            "source": "integration_test",
        },
    )

    fact = manager.memory_modules["semantic"]._store.get_many([memory_id])[0]
    assert fact.concepts
    assert fact.metadata.get("concept_extraction_source") in {"metadata", "llm"}

    results = manager.search_memory(
        query="PostgreSQL Neo4j 记忆",
        memory_type="semantic",
        limit=5,
        session_id="semantic_integration_session",
    )

    assert any(record.id == memory_id for record in results)

    manager.remove_memory(memory_id, "semantic")


@pytest.mark.skipif(
    os.getenv("RUN_SEMANTIC_INTEGRATION") != "1",
    reason="需要本地 Neo4j、Milvus、EMBED_API_KEY 与 LLM",
)
def test_semantic_memory_llm_concept_extraction() -> None:
    config = MemoryConfig.from_env()
    if not config.llm_api_key:
        pytest.skip("未配置 LLM_API_KEY")
    if not config.llm_base_url:
        pytest.skip("未配置 LLM_BASE_URL 或 EMBED_BASE_URL")
    if not config.embed_api_key or not config.neo4j_uri:
        pytest.skip("未配置 EMBED_API_KEY 或 NEO4J_URI")

    manager = MemoryManager(
        config=config,
        user_id="semantic_llm_integration_user",
        enable_working=False,
        enable_episodic=False,
        enable_semantic=True,
        enable_perceptual=False,
    )

    memory_id = manager.add_memory(
        content="用户长期规则：回答必须简洁，技术讨论优先引用官方文档。",
        memory_type="semantic",
        importance=0.9,
        metadata={"session_id": "llm_concept_session"},
    )

    fact = manager.memory_modules["semantic"]._store.get_many([memory_id])[0]
    assert fact.metadata.get("concept_extraction_source") == "llm"
    assert len(fact.concepts) >= 1

    manager.remove_memory(memory_id, "semantic")
