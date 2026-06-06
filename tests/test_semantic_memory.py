"""SemanticMemory tests with fake Neo4j/Milvus backends."""

from __future__ import annotations

import pytest

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.modules import MemoryRecord, SemanticMemory
from tests.concept_fakes import StubConceptExtractor
from tests.semantic_fakes import (
    create_semantic_bundle,
    create_semantic_memory_with_outbox,
)


def _flush(memory: SemanticMemory) -> None:
    memory.flush_vector_outbox()


def test_semantic_memory_add_writes_graph_and_vector_stores() -> None:
    bundle = create_semantic_bundle()
    memory, _ = create_semantic_memory_with_outbox(
        bundle,
        concept_extractor=StubConceptExtractor(),
    )

    memory_id = memory.add(
        content="用户喜欢 Python 和机器学习",
        importance=0.8,
        metadata={"concepts": ["Python", "机器学习"], "source": "chat"},
    )
    _flush(memory)

    assert memory_id in bundle.store.facts
    assert memory_id in bundle.vectors.records
    assert bundle.embeddings.calls == [
        "用户喜欢 Python 和机器学习",
    ]
    fact = bundle.store.facts[memory_id]
    assert fact.user_id == "user123"
    assert fact.concepts == ["Python", "机器学习"]
    assert fact.metadata["source"] == "chat"
    assert fact.metadata["concept_extraction_source"] == "metadata"
    assert fact.embedding_status == "done"


def test_semantic_memory_add_uses_llm_concepts() -> None:
    bundle = create_semantic_bundle()
    memory, _ = create_semantic_memory_with_outbox(
        bundle,
        concept_extractor=StubConceptExtractor(
            llm_concepts=["长期规则", "回答简洁"],
        ),
    )

    memory_id = memory.add(
        content="长期规则：回答要简洁",
        importance=0.7,
        metadata={},
    )
    _flush(memory)

    assert bundle.store.facts[memory_id].concepts == ["长期规则", "回答简洁"]
    assert bundle.store.facts[memory_id].metadata["concept_extraction_source"] == "llm"


def test_semantic_memory_retrieve_uses_vector_graph_and_importance_formula() -> None:
    """RRF 融合：向量榜 + 图榜；同 importance 时 graph 分更高者靠前。"""
    bundle = create_semantic_bundle()
    memory, _ = create_semantic_memory_with_outbox(
        bundle,
        concept_extractor=StubConceptExtractor(query_concepts=["Python", "机器学习"]),
    )

    python_id = memory.add(
        "用户偏好 Python 机器学习",
        0.6,
        {"concepts": ["Python", "机器学习"]},
    )
    react_id = memory.add(
        "用户偏好 React 前端开发",
        0.6,
        {"concepts": ["React", "前端"]},
    )
    _flush(memory)
    bundle.store.graph_scores[python_id] = 1.0
    bundle.store.graph_scores[react_id] = 0.0
    bundle.vectors.records[react_id]["vector"] = list(
        bundle.vectors.records[python_id]["vector"]
    )

    results = memory.retrieve("Python 机器学习", limit=2)

    assert [record.id for record in results] == [python_id, react_id]
    assert all(isinstance(record, MemoryRecord) for record in results)


def test_semantic_memory_retrieve_read_your_writes_includes_pending() -> None:
    bundle = create_semantic_bundle()
    memory, _ = create_semantic_memory_with_outbox(
        bundle,
        concept_extractor=StubConceptExtractor(query_concepts=["Neo4j"]),
    )

    memory_id = memory.add(
        "用户偏好 Neo4j outbox",
        0.8,
        {"concepts": ["Neo4j", "outbox"]},
    )
    assert memory_id not in bundle.vectors.records

    results = memory.retrieve("Neo4j outbox", limit=3)
    assert any(record.id == memory_id for record in results)


def test_memory_manager_semantic_uses_injected_backends() -> None:
    bundle = create_semantic_bundle()
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=False,
        enable_episodic=False,
        enable_semantic=True,
        semantic_store=bundle.store,
        semantic_vector_store=bundle.vectors,
        semantic_embedding_provider=bundle.embeddings,
        concept_extractor=StubConceptExtractor(query_concepts=["中文回答"]),
    )

    memory_id = manager.add_memory(
        content="用户偏好中文回答",
        memory_type="semantic",
        importance=0.9,
        metadata={"concepts": ["用户偏好", "中文回答"]},
    )
    manager.flush_vector_outbox()

    assert memory_id in bundle.store.facts
    results = manager.search_memory(
        query="中文回答",
        memory_type="semantic",
        limit=3,
    )
    assert len(results) >= 1


def test_semantic_memory_update_preserves_id_and_reindexes_vector() -> None:
    bundle = create_semantic_bundle()
    memory, _ = create_semantic_memory_with_outbox(
        bundle,
        concept_extractor=StubConceptExtractor(),
    )

    memory_id = memory.add(
        "用户偏好 PostgreSQL",
        0.7,
        {"concepts": ["PostgreSQL"], "session_id": "s1"},
    )
    _flush(memory)
    assert memory_id in bundle.vectors.records

    updated_id = memory.update(
        memory_id,
        content="用户偏好 PostgreSQL 与 Neo4j",
        importance=0.9,
        metadata={"concepts": ["PostgreSQL", "Neo4j"], "session_id": "s1"},
    )
    _flush(memory)

    assert updated_id == memory_id
    assert len([fact for fact in bundle.store.facts.values() if not fact.deleted]) == 1
    fact = bundle.store.facts[memory_id]
    assert fact.content == "用户偏好 PostgreSQL 与 Neo4j"
    assert fact.importance == 0.9
    assert "Neo4j" in fact.concepts
    assert memory_id in bundle.vectors.records
    assert bundle.embeddings.calls[-1] == "用户偏好 PostgreSQL 与 Neo4j"
    assert fact.version == 2


def test_semantic_memory_update_raises_when_missing() -> None:
    bundle = create_semantic_bundle()
    memory, _ = create_semantic_memory_with_outbox(
        bundle,
        concept_extractor=StubConceptExtractor(),
    )

    with pytest.raises(KeyError, match="未找到记忆"):
        memory.update(
            "missing-id",
            content="x",
            importance=0.5,
            metadata={},
        )


def test_semantic_neo4j_outbox_version_mismatch_is_superseded() -> None:
    bundle = create_semantic_bundle()
    memory, processor = create_semantic_memory_with_outbox(
        bundle,
        concept_extractor=StubConceptExtractor(),
    )

    memory_id = memory.add("v1", 0.5, {"concepts": ["a"]})
    stale_event = next(iter(bundle.store.outbox_events.values()))
    memory.update(memory_id, content="v2", importance=0.6, metadata={"concepts": ["a"]})

    bundle.store.outbox_events[stale_event.event_id].status = "pending"
    bundle.store.outbox_events[stale_event.event_id].attempts = 0

    ok, failed = processor.process_batch(batch_size=10)
    assert ok >= 1
    assert failed == 0
    assert bundle.store.outbox_events[stale_event.event_id].status == "superseded"
