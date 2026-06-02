"""MemoryManager tests."""

import pytest

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.modules import EpisodicMemory, PerceptualMemory, SemanticMemory, WorkingMemory
from tests.episodic_fakes import (
    FakeEmbeddingProvider,
    FakeEpisodicStore,
    FakeVectorStore,
)
from tests.semantic_fakes import create_semantic_bundle


def test_memory_manager_initializes_enabled_memory_modules() -> None:
    semantic_bundle = create_semantic_bundle()
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=True,
        enable_episodic=False,
        enable_semantic=True,
        enable_perceptual=False,
        semantic_store=semantic_bundle.store,
        semantic_vector_store=semantic_bundle.vectors,
        semantic_embedding_provider=semantic_bundle.embeddings,
    )

    assert set(manager.memory_modules) == {"working", "semantic"}
    assert isinstance(manager.memory_modules["working"], WorkingMemory)
    assert isinstance(manager.memory_modules["semantic"], SemanticMemory)


def test_memory_manager_can_initialize_all_memory_modules() -> None:
    semantic_bundle = create_semantic_bundle()
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=True,
        enable_episodic=True,
        enable_semantic=True,
        enable_perceptual=True,
        episodic_store=FakeEpisodicStore(),
        vector_store=FakeVectorStore(),
        embedding_provider=FakeEmbeddingProvider(vector_size=8),
        semantic_store=semantic_bundle.store,
        semantic_vector_store=semantic_bundle.vectors,
        semantic_embedding_provider=semantic_bundle.embeddings,
    )

    assert isinstance(manager.memory_modules["working"], WorkingMemory)
    assert isinstance(manager.memory_modules["episodic"], EpisodicMemory)
    assert isinstance(manager.memory_modules["semantic"], SemanticMemory)
    assert isinstance(manager.memory_modules["perceptual"], PerceptualMemory)


def test_memory_manager_add_memory_delegates_to_selected_module() -> None:
    semantic_bundle = create_semantic_bundle()
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=False,
        enable_episodic=False,
        enable_semantic=True,
        enable_perceptual=False,
        semantic_store=semantic_bundle.store,
        semantic_vector_store=semantic_bundle.vectors,
        semantic_embedding_provider=semantic_bundle.embeddings,
    )

    memory_id = manager.add_memory(
        content="用户喜欢 Python",
        memory_type="semantic",
        importance=0.8,
        metadata={"source": "chat"},
    )

    fact = semantic_bundle.store.facts[memory_id]
    assert fact.content == "用户喜欢 Python"
    assert fact.importance == 0.8
    assert fact.metadata == {"source": "chat"}


def test_memory_manager_rejects_disabled_memory_type() -> None:
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=True,
        enable_episodic=False,
        enable_semantic=False,
        enable_perceptual=False,
    )

    with pytest.raises(ValueError, match="未启用记忆类型"):
        manager.add_memory(
            content="用户喜欢 Python",
            memory_type="semantic",
            importance=0.8,
            metadata={},
        )
