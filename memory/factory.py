"""记忆模块工厂：从 MemoryManager.__init__ 抽取的创建逻辑。"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig
from .embedding import EmbeddingProvider, create_embedding_provider
from .modules import (
    EpisodicMemory,
    PerceptualMemory,
    SemanticMemory,
)
from .storage.document_store import PerceptualMemoryStore, create_perceptual_store
from .storage.milvus_store import MilvusVectorStore, create_vector_store
from .storage.neo4j_store import SemanticMemoryStore, create_semantic_store
from .storage.postgres_outbox_store import (
    PostgresVectorOutboxStore,
    create_postgres_outbox_store,
)
from .storage.postgres_store import EpisodicMemoryStore, create_episodic_store
from .concept_extractor import ConceptExtractor
from .semantic_outbox_processor import SemanticOutboxProcessor
from .storage.vector_outbox import VectorOutbox
from .vector_outbox_processor import VectorOutboxProcessor


def setup_outbox(
    config: MemoryConfig,
) -> tuple[PostgresVectorOutboxStore | None, VectorOutbox | None, VectorOutboxProcessor | None]:
    """根据配置初始化 outbox 基础设施。

    返回 (pg_outbox, inmem_outbox, processor)；无 outbox 时三项均为 None。
    """
    if (
        config.enable_persistent_vector_outbox
        and config.enable_vector_outbox
        and config.database_url
    ):
        pg_outbox = create_postgres_outbox_store(config)
        processor = VectorOutboxProcessor(config, pg_outbox)
        return pg_outbox, None, processor

    if config.enable_vector_outbox:
        inmem_outbox = VectorOutbox(max_attempts=config.vector_outbox_max_attempts)
        return None, inmem_outbox, None

    return None, None, None


def create_episodic_memory(
    config: MemoryConfig,
    user_id: str,
    *,
    episodic_store: EpisodicMemoryStore | None = None,
    vector_store: MilvusVectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_outbox: VectorOutbox | None = None,
    pg_vector_outbox: PostgresVectorOutboxStore | None = None,
    outbox_processor: VectorOutboxProcessor | None = None,
) -> EpisodicMemory:
    store = episodic_store or create_episodic_store(config)
    vectors = vector_store or create_vector_store(config)
    embeddings = embedding_provider or create_embedding_provider(config)
    return EpisodicMemory(
        config=config,
        user_id=user_id,
        episodic_store=store,
        vector_store=vectors,
        embedding_provider=embeddings,
        vector_outbox=vector_outbox,
        pg_vector_outbox=pg_vector_outbox,
        outbox_processor=outbox_processor,
    )


def create_semantic_memory(
    config: MemoryConfig,
    user_id: str,
    *,
    semantic_store: SemanticMemoryStore | None = None,
    vector_store: MilvusVectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    concept_extractor: ConceptExtractor | None = None,
) -> SemanticMemory:
    store = semantic_store or create_semantic_store(config)
    vectors = vector_store or create_vector_store(
        config,
        collection_name=config.semantic_milvus_collection(),
    )
    embeddings = embedding_provider or create_embedding_provider(config)
    if not hasattr(store, "claim_pending_outbox_events"):
        raise ValueError("语义记忆 store 需支持 Neo4j Transactional Outbox")
    semantic_outbox_processor = SemanticOutboxProcessor(
        config,
        store,
        embedding_provider=embeddings,
        vector_store=vectors,
    )
    return SemanticMemory(
        config=config,
        user_id=user_id,
        semantic_store=store,
        vector_store=vectors,
        embedding_provider=embeddings,
        concept_extractor=concept_extractor,
        semantic_outbox_processor=semantic_outbox_processor,
    )


def create_perceptual_memory(
    config: MemoryConfig,
    user_id: str,
    *,
    perceptual_store: PerceptualMemoryStore | None = None,
    vector_stores: dict[str, MilvusVectorStore] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    pg_vector_outbox: PostgresVectorOutboxStore | None = None,
    outbox_processor: VectorOutboxProcessor | None = None,
) -> PerceptualMemory:
    store = perceptual_store or create_perceptual_store()
    vectors = vector_stores or {
        modality: create_vector_store(
            config,
            collection_name=config.perceptual_milvus_collection(modality),
        )
        for modality in ("text", "image", "audio", "video", "file")
    }
    embeddings = embedding_provider or create_embedding_provider(config)
    return PerceptualMemory(
        config=config,
        user_id=user_id,
        perceptual_store=store,
        vector_stores=vectors,
        embedding_provider=embeddings,
        pg_vector_outbox=pg_vector_outbox,
        outbox_processor=outbox_processor,
    )
