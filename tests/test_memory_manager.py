"""MemoryManager tests."""

import pytest

from memory.config import MemoryConfig
from memory.concept_extractor import ConceptExtractor
from memory.manager import MemoryManager
from memory.modules import EpisodicMemory, PerceptualMemory, SemanticMemory, WorkingMemory
from tests.concept_fakes import StubConceptExtractor
from tests.episodic_fakes import (
    FakeEmbeddingProvider,
    FakeEpisodicStore,
    FakeVectorStore,
)
from tests.perceptual_fakes import create_perceptual_bundle
from tests.semantic_fakes import create_semantic_bundle


def _stub_extractor() -> ConceptExtractor:
    return StubConceptExtractor(llm_concepts=["Python", "用户喜欢"])


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
        concept_extractor=_stub_extractor(),
    )

    assert set(manager.memory_modules) == {"working", "semantic"}
    assert isinstance(manager.memory_modules["working"], WorkingMemory)
    assert isinstance(manager.memory_modules["semantic"], SemanticMemory)


def test_memory_manager_can_initialize_all_memory_modules() -> None:
    semantic_bundle = create_semantic_bundle()
    perceptual_bundle = create_perceptual_bundle()
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
        perceptual_vector_stores=perceptual_bundle.vector_stores,
        perceptual_embedding_provider=perceptual_bundle.embeddings,
        concept_extractor=_stub_extractor(),
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
        concept_extractor=_stub_extractor(),
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
    assert fact.metadata["source"] == "chat"
    assert fact.metadata["concept_extraction_source"] == "llm"


def test_memory_manager_semantic_update_preserves_memory_id() -> None:
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
        concept_extractor=StubConceptExtractor(),
    )

    memory_id = manager.add_memory(
        content="旧规则：回答要简短",
        memory_type="semantic",
        importance=0.6,
        metadata={"concepts": ["回答简短"]},
    )

    returned_id = manager.update_memory(
        memory_id,
        "semantic",
        content="新规则：回答要简洁且引用来源",
        importance=0.85,
    )

    assert returned_id == memory_id
    assert len(semantic_bundle.store.facts) == 1
    assert semantic_bundle.store.facts[memory_id].content == "新规则：回答要简洁且引用来源"
    assert semantic_bundle.store.facts[memory_id].importance == 0.85


def test_memory_manager_episodic_update_preserves_memory_id() -> None:
    store = FakeEpisodicStore()
    vectors = FakeVectorStore()
    embeddings = FakeEmbeddingProvider(vector_size=8)
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=False,
        enable_episodic=True,
        enable_semantic=False,
        episodic_store=store,
        vector_store=vectors,
        embedding_provider=embeddings,
    )

    memory_id = manager.add_memory(
        content="第一次对话摘要",
        memory_type="episodic",
        importance=0.6,
        metadata={"session_id": "s1"},
    )
    sequence_no = store.events[memory_id].sequence_no

    returned_id = manager.update_memory(
        memory_id,
        "episodic",
        content="更新后的对话摘要",
        importance=0.8,
    )

    assert returned_id == memory_id
    assert store.events[memory_id].sequence_no == sequence_no
    assert store.events[memory_id].content == "更新后的对话摘要"


def test_memory_manager_perceptual_update_preserves_memory_id() -> None:
    perceptual_bundle = create_perceptual_bundle()
    from memory.storage.document_store import InMemoryPerceptualStore

    store = InMemoryPerceptualStore()
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=False,
        enable_episodic=False,
        enable_semantic=False,
        enable_perceptual=True,
        perceptual_store=store,
        perceptual_vector_stores=perceptual_bundle.vector_stores,
        perceptual_embedding_provider=perceptual_bundle.embeddings,
    )

    memory_id = manager.add_memory(
        content="白板照片",
        memory_type="perceptual",
        importance=0.6,
        metadata={"modality": "image", "caption": "旧白板"},
    )

    returned_id = manager.update_memory(
        memory_id,
        "perceptual",
        content="白板照片（已更新）",
        importance=0.9,
        metadata={"modality": "image", "caption": "新白板"},
    )

    assert returned_id == memory_id
    assert store.get(memory_id).content == "白板照片（已更新）"
    assert memory_id in perceptual_bundle.image_vectors.records


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
