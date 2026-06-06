"""PerceptualMemory tests with fake multi-modal vector backends."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.modules import MemoryRecord, PerceptualMemory
from memory.storage.document_store import InMemoryPerceptualStore
from tests.perceptual_fakes import create_perceptual_bundle


def test_perceptual_memory_add_routes_image_to_image_collection() -> None:
    bundle = create_perceptual_bundle()
    store = InMemoryPerceptualStore()
    memory = PerceptualMemory(
        config=MemoryConfig(),
        user_id="user123",
        perceptual_store=store,
        vector_stores=bundle.vector_stores,
        embedding_provider=bundle.embeddings,
    )

    memory_id = memory.add(
        content="用户上传了一张架构图",
        importance=0.8,
        metadata={
            "modality": "image",
            "raw_data": "/tmp/architecture.png",
            "caption": "系统架构图",
            "session_id": "session_1",
        },
    )

    item = store.get(memory_id)
    assert item is not None
    assert item.modality == "image"
    assert item.raw_data == "/tmp/architecture.png"
    assert memory_id in bundle.image_vectors.records
    assert memory_id not in bundle.text_vectors.records
    assert bundle.embeddings.calls == ["系统架构图"]


def test_perceptual_memory_add_uses_transcript_for_audio_embedding() -> None:
    bundle = create_perceptual_bundle()
    memory = PerceptualMemory(
        config=MemoryConfig(),
        user_id="user123",
        perceptual_store=InMemoryPerceptualStore(),
        vector_stores=bundle.vector_stores,
        embedding_provider=bundle.embeddings,
    )

    memory_id = memory.add(
        content="用户上传了语音片段",
        importance=0.7,
        metadata={
            "modality": "audio",
            "raw_data": "/tmp/voice.wav",
            "transcript": "提醒我明天检查 Milvus",
        },
    )

    assert memory_id in bundle.audio_vectors.records
    assert bundle.embeddings.calls == ["提醒我明天检查 Milvus"]


def test_perceptual_memory_retrieve_uses_vector_recency_and_importance_formula() -> None:
    bundle = create_perceptual_bundle()
    store = InMemoryPerceptualStore()
    memory = PerceptualMemory(
        config=MemoryConfig(),
        user_id="user123",
        perceptual_store=store,
        vector_stores=bundle.vector_stores,
        embedding_provider=bundle.embeddings,
    )
    now = datetime.now()

    fresh_id = memory.add(
        "用户上传了新的架构图",
        0.6,
        {
            "modality": "image",
            "caption": "Milvus 和 Neo4j 架构",
            "timestamp": now.isoformat(),
        },
    )
    old_id = memory.add(
        "用户上传了旧的架构图",
        1.0,
        {
            "modality": "image",
            "caption": "Milvus 和 Neo4j 架构",
            "timestamp": (now - timedelta(days=30)).isoformat(),
        },
    )
    bundle.image_vectors.records[old_id]["vector"] = list(
        bundle.image_vectors.records[fresh_id]["vector"]
    )

    results = memory.retrieve("Milvus Neo4j 架构", limit=2, modality="image")

    assert [record.id for record in results] == [fresh_id, old_id]
    assert all(isinstance(record, MemoryRecord) for record in results)


def test_perceptual_memory_cross_modal_retrieve_searches_all_modalities() -> None:
    bundle = create_perceptual_bundle()
    memory = PerceptualMemory(
        config=MemoryConfig(),
        user_id="user123",
        perceptual_store=InMemoryPerceptualStore(),
        vector_stores=bundle.vector_stores,
        embedding_provider=bundle.embeddings,
    )

    text_id = memory.add(
        "用户说需要保存数据库架构",
        0.5,
        {"modality": "text"},
    )
    image_id = memory.add(
        "用户上传了一张数据库架构图",
        0.8,
        {"modality": "image", "caption": "数据库架构"},
    )

    results = memory.retrieve("数据库架构", limit=5)
    result_ids = {record.id for record in results}

    assert text_id in result_ids
    assert image_id in result_ids


def test_memory_manager_perceptual_uses_injected_backends() -> None:
    bundle = create_perceptual_bundle()
    store = InMemoryPerceptualStore()
    manager = MemoryManager(
        config=MemoryConfig(),
        user_id="user123",
        enable_working=False,
        enable_episodic=False,
        enable_semantic=False,
        enable_perceptual=True,
        perceptual_store=store,
        perceptual_vector_stores=bundle.vector_stores,
        perceptual_embedding_provider=bundle.embeddings,
    )

    memory_id = manager.add_memory(
        content="用户上传了白板照片",
        memory_type="perceptual",
        importance=0.7,
        metadata={"modality": "image", "caption": "白板任务列表"},
    )

    assert store.get(memory_id) is not None
    results = manager.search_memory(
        query="白板任务",
        memory_type="perceptual",
        limit=3,
    )
    assert len(results) >= 1


def test_perceptual_memory_rejects_invalid_modality() -> None:
    bundle = create_perceptual_bundle()
    memory = PerceptualMemory(
        config=MemoryConfig(),
        user_id="user123",
        perceptual_store=InMemoryPerceptualStore(),
        vector_stores=bundle.vector_stores,
        embedding_provider=bundle.embeddings,
    )

    with pytest.raises(ValueError, match="不支持的感知模态"):
        memory.add("内容", 0.5, {"modality": "hologram"})


def test_perceptual_memory_update_preserves_id_and_reindexes_vector() -> None:
    bundle = create_perceptual_bundle()
    store = InMemoryPerceptualStore()
    memory = PerceptualMemory(
        config=MemoryConfig(),
        user_id="user123",
        perceptual_store=store,
        vector_stores=bundle.vector_stores,
        embedding_provider=bundle.embeddings,
    )

    memory_id = memory.add(
        "用户上传了架构图",
        0.7,
        {"modality": "image", "caption": "旧架构图", "session_id": "s1"},
    )
    assert memory_id in bundle.image_vectors.records

    updated_id = memory.update(
        memory_id,
        content="用户上传了新版架构图",
        importance=0.9,
        metadata={"modality": "image", "caption": "新架构图", "session_id": "s1"},
    )

    assert updated_id == memory_id
    item = store.get(memory_id)
    assert item is not None
    assert item.content == "用户上传了新版架构图"
    assert item.importance == 0.9
    assert memory_id in bundle.image_vectors.records
    assert bundle.embeddings.calls[-1] == "新架构图"


def test_perceptual_memory_update_switches_modality_and_moves_vector() -> None:
    bundle = create_perceptual_bundle()
    store = InMemoryPerceptualStore()
    memory = PerceptualMemory(
        config=MemoryConfig(),
        user_id="user123",
        perceptual_store=store,
        vector_stores=bundle.vector_stores,
        embedding_provider=bundle.embeddings,
    )

    memory_id = memory.add(
        "用户说需要保存数据库架构",
        0.6,
        {"modality": "text", "session_id": "s1"},
    )
    assert memory_id in bundle.text_vectors.records
    assert memory_id not in bundle.image_vectors.records

    memory.update(
        memory_id,
        content="用户上传了数据库架构图",
        importance=0.8,
        metadata={"modality": "image", "caption": "数据库架构", "session_id": "s1"},
    )

    assert store.get(memory_id).modality == "image"
    assert memory_id not in bundle.text_vectors.records
    assert memory_id in bundle.image_vectors.records


def test_perceptual_memory_update_raises_when_missing() -> None:
    bundle = create_perceptual_bundle()
    memory = PerceptualMemory(
        config=MemoryConfig(),
        user_id="user123",
        perceptual_store=InMemoryPerceptualStore(),
        vector_stores=bundle.vector_stores,
        embedding_provider=bundle.embeddings,
    )

    with pytest.raises(KeyError, match="未找到记忆"):
        memory.update(
            "missing-id",
            content="内容",
            importance=0.5,
            metadata={"modality": "text"},
        )
