"""记忆管理器：组装模块并委托操作到 MemoryOperations mixin。"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig
from .embedding import EmbeddingProvider
from .factory import (
    create_episodic_memory,
    create_perceptual_memory,
    create_semantic_memory,
    setup_outbox,
)
from .modules import InMemoryStore, WorkingMemory
from .modules.base import MemoryRecord
from .operations import MemoryOperations
from .storage.document_store import PerceptualMemoryStore
from .storage.milvus_store import MilvusVectorStore
from .storage.neo4j_store import SemanticMemoryStore
from .storage.postgres_outbox_store import PostgresVectorOutboxStore
from .storage.postgres_store import EpisodicMemoryStore
from .concept_extractor import ConceptExtractor
from .storage.vector_outbox import VectorOutbox
from .vector_outbox_processor import VectorOutboxProcessor

__all__ = [
    "MemoryManager",
    "MemoryRecord",
    "EpisodicMemoryStore",
    "MilvusVectorStore",
    "SemanticMemoryStore",
    "PerceptualMemoryStore",
    "PostgresVectorOutboxStore",
    "VectorOutbox",
    "VectorOutboxProcessor",
    "EmbeddingProvider",
    "ConceptExtractor",
]


class MemoryManager(MemoryOperations):
    """管理不同类型的记忆模块。

    工厂逻辑委托到 memory.factory，CRUD/forget/consolidate/stats
    委托到 MemoryOperations mixin。
    """

    def __init__(
        self,
        config: MemoryConfig,
        user_id: str,
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False,
        episodic_store: EpisodicMemoryStore | None = None,
        vector_store: MilvusVectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_store: SemanticMemoryStore | None = None,
        semantic_vector_store: MilvusVectorStore | None = None,
        semantic_embedding_provider: EmbeddingProvider | None = None,
        perceptual_store: PerceptualMemoryStore | None = None,
        perceptual_vector_stores: dict[str, MilvusVectorStore] | None = None,
        perceptual_embedding_provider: EmbeddingProvider | None = None,
        concept_extractor: ConceptExtractor | None = None,
    ) -> None:
        self.config = config
        self.user_id = user_id
        self.enable_working = enable_working
        self.enable_episodic = enable_episodic
        self.enable_semantic = enable_semantic
        self.enable_perceptual = enable_perceptual
        self._store = InMemoryStore()
        self.memory_modules: dict[str, Any] = {}
        self._concept_extractor = concept_extractor

        # Outbox setup
        self.pg_vector_outbox, self.vector_outbox, self._outbox_processor = setup_outbox(config)

        # Module creation via factory
        if enable_working:
            self.memory_modules["working"] = WorkingMemory(self.config, self._store)
        if enable_episodic:
            self.memory_modules["episodic"] = create_episodic_memory(
                self.config,
                self.user_id,
                episodic_store=episodic_store,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
                vector_outbox=self.vector_outbox,
                pg_vector_outbox=self.pg_vector_outbox,
                outbox_processor=self._outbox_processor,
            )
        if enable_semantic:
            self.memory_modules["semantic"] = create_semantic_memory(
                self.config,
                self.user_id,
                semantic_store=semantic_store,
                vector_store=semantic_vector_store,
                embedding_provider=semantic_embedding_provider,
                concept_extractor=self._concept_extractor,
            )
        if enable_perceptual:
            self.memory_modules["perceptual"] = create_perceptual_memory(
                self.config,
                self.user_id,
                perceptual_store=perceptual_store,
                vector_stores=perceptual_vector_stores,
                embedding_provider=perceptual_embedding_provider,
                pg_vector_outbox=self.pg_vector_outbox,
                outbox_processor=self._outbox_processor,
            )

        # Wire outbox processor to episodic store
        if self._outbox_processor is not None and "episodic" in self.memory_modules:
            episodic_module = self.memory_modules["episodic"]
            store = getattr(episodic_module, "_store", None)
            if store is not None and hasattr(store, "mark_vector_indexed"):
                self._outbox_processor._episodic_store = store
