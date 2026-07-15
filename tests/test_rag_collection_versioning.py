"""RAG Milvus collection versioning tests (M-04)."""

from __future__ import annotations

from rag.config import RagConfig
from memory.collection_names import versioned_collection_name


def test_rag_config_versioned_collection_default() -> None:
    """默认启用版本化时 collection_name 包含 model slug 和 vector_size。"""
    config = RagConfig()

    resolved = config.rag_milvus_collection()

    assert resolved != config.collection_name
    assert "text_embedding_v3" in resolved
    assert str(config.vector_size) in resolved


def test_rag_config_versioned_collection_disabled() -> None:
    """关闭版本化时返回原始 collection_name。"""
    config = RagConfig(use_versioned_milvus_collections=False)

    assert config.rag_milvus_collection() == config.collection_name


def test_rag_config_versioned_collection_custom_model() -> None:
    """自定义 embedding model 名反映在 collection_name 中。"""
    config = RagConfig(
        embed_model_name="bge-large-zh",
        vector_size=1024,
    )

    resolved = config.rag_milvus_collection()
    expected = versioned_collection_name(
        config.collection_name, "bge-large-zh", 1024
    )
    assert resolved == expected
    assert "bge_large_zh" in resolved


def test_rag_config_from_env_parses_versioned_flag() -> None:
    """from_env 解析 USE_VERSIONED_MILVUS_COLLECTIONS。"""
    import os

    old = os.environ.get("USE_VERSIONED_MILVUS_COLLECTIONS")
    os.environ["USE_VERSIONED_MILVUS_COLLECTIONS"] = "false"
    try:
        config = RagConfig.from_env()
        assert config.use_versioned_milvus_collections is False
        assert config.rag_milvus_collection() == config.collection_name
    finally:
        if old is None:
            os.environ.pop("USE_VERSIONED_MILVUS_COLLECTIONS", None)
        else:
            os.environ["USE_VERSIONED_MILVUS_COLLECTIONS"] = old


def test_rag_config_from_env_parses_embed_model_name() -> None:
    """from_env 解析 EMBED_MODEL_NAME。"""
    import os

    old = os.environ.get("EMBED_MODEL_NAME")
    os.environ["EMBED_MODEL_NAME"] = "custom-embed-v2"
    try:
        config = RagConfig.from_env()
        assert config.embed_model_name == "custom-embed-v2"
        resolved = config.rag_milvus_collection()
        assert "custom_embed_v2" in resolved
    finally:
        if old is None:
            os.environ.pop("EMBED_MODEL_NAME", None)
        else:
            os.environ["EMBED_MODEL_NAME"] = old


def test_rag_manager_uses_versioned_collection() -> None:
    """RagManager 实际使用版本化后的 collection_name。"""
    from rag.chunker import MarkdownChunker
    from rag.manager import RagManager
    from rag.storage import InMemoryRagStore
    from tests.rag_fakes import FakeConverter, FakeEmbeddingProvider, FakeLLM, FakeVectorStore

    config = RagConfig(
        enable_rag_vector_outbox=False,
        database_url=None,
        embed_model_name="test-model-v1",
        vector_size=8,
    )
    vector_store = FakeVectorStore()
    manager = RagManager(
        config=config,
        store=InMemoryRagStore(),
        converter=FakeConverter("# Test\n\ncontent"),
        chunker=MarkdownChunker(target_tokens=20, max_tokens=40, overlap_tokens=5),
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(vector_size=8),
        llm=FakeLLM(),
    )

    stats = manager.stats()

    assert "test_model_v1" in stats["collection"]
    assert "8" in stats["collection"]
