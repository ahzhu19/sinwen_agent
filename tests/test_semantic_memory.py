"""SemanticMemory tests with fake Neo4j/Milvus backends."""

from __future__ import annotations

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.modules import MemoryRecord, SemanticMemory
from tests.semantic_fakes import create_semantic_bundle


def test_semantic_memory_add_writes_graph_and_vector_stores() -> None:
    bundle = create_semantic_bundle()
    memory = SemanticMemory(
        config=MemoryConfig(),
        user_id="user123",
        semantic_store=bundle.store,
        vector_store=bundle.vectors,
        embedding_provider=bundle.embeddings,
    )

    memory_id = memory.add(
        content="用户喜欢 Python 和机器学习",
        importance=0.8,
        metadata={"concepts": ["Python", "机器学习"], "source": "chat"},
    )

    assert memory_id in bundle.store.facts
    assert memory_id in bundle.vectors.records
    assert bundle.embeddings.calls == ["用户喜欢 Python 和机器学习"]
    fact = bundle.store.facts[memory_id]
    assert fact.user_id == "user123"
    assert fact.concepts == ["Python", "机器学习"]
    assert fact.metadata["source"] == "chat"


def test_semantic_memory_add_uses_simple_concept_fallback() -> None:
    bundle = create_semantic_bundle()
    memory = SemanticMemory(
        config=MemoryConfig(),
        user_id="user123",
        semantic_store=bundle.store,
        vector_store=bundle.vectors,
        embedding_provider=bundle.embeddings,
    )

    memory_id = memory.add(
        content="长期规则：回答要简洁",
        importance=0.7,
        metadata={},
    )

    assert bundle.store.facts[memory_id].concepts
    assert "长期规则" in bundle.store.facts[memory_id].concepts


def test_semantic_memory_retrieve_uses_vector_graph_and_importance_formula() -> None:
    bundle = create_semantic_bundle()
    memory = SemanticMemory(
        config=MemoryConfig(),
        user_id="user123",
        semantic_store=bundle.store,
        vector_store=bundle.vectors,
        embedding_provider=bundle.embeddings,
    )

    python_id = memory.add(
        "用户偏好 Python 机器学习",
        0.6,
        {"concepts": ["Python", "机器学习"]},
    )
    react_id = memory.add(
        "用户偏好 React 前端开发",
        1.0,
        {"concepts": ["React", "前端"]},
    )
    bundle.store.graph_scores[python_id] = 1.0
    bundle.store.graph_scores[react_id] = 0.0
    bundle.vectors.records[react_id]["vector"] = list(bundle.vectors.records[python_id]["vector"])

    results = memory.retrieve("Python 机器学习", limit=2)

    assert [record.id for record in results] == [python_id, react_id]
    assert all(isinstance(record, MemoryRecord) for record in results)


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
    )

    memory_id = manager.add_memory(
        content="用户偏好中文回答",
        memory_type="semantic",
        importance=0.9,
        metadata={"concepts": ["用户偏好", "中文回答"]},
    )

    assert memory_id in bundle.store.facts
    results = manager.search_memory(
        query="中文回答",
        memory_type="semantic",
        limit=3,
    )
    assert len(results) >= 1
