"""记忆管理器：承载核心记忆管理逻辑。"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig
from .embedding import EmbeddingProvider, create_embedding_provider
from .episodic_store import EpisodicMemoryStore, create_episodic_store
from .semantic_store import SemanticMemoryStore, create_semantic_store
from .modules import (
    EpisodicMemory,
    InMemoryStore,
    PerceptualMemory,
    SemanticMemory,
    WorkingMemory,
)
from .vector_store import MilvusVectorStore, create_vector_store


class MemoryManager:
    """管理不同类型的记忆模块。"""

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
    ) -> None:
        self.config = config
        self.user_id = user_id
        self.enable_working = enable_working
        self.enable_episodic = enable_episodic
        self.enable_semantic = enable_semantic
        self.enable_perceptual = enable_perceptual
        self.store = InMemoryStore()
        self.memory_modules: dict[str, Any] = {}

        if enable_working:
            self.memory_modules["working"] = WorkingMemory(self.config, self.store)
        if enable_episodic:
            self.memory_modules["episodic"] = self._create_episodic_memory(
                episodic_store=episodic_store,
                vector_store=vector_store,
                embedding_provider=embedding_provider,
            )
        if enable_semantic:
            self.memory_modules["semantic"] = self._create_semantic_memory(
                semantic_store=semantic_store,
                vector_store=semantic_vector_store,
                embedding_provider=semantic_embedding_provider,
            )
        if enable_perceptual:
            self.memory_modules["perceptual"] = PerceptualMemory(self.config, self.store)

    def _create_episodic_memory(
        self,
        episodic_store: EpisodicMemoryStore | None,
        vector_store: MilvusVectorStore | None,
        embedding_provider: EmbeddingProvider | None,
    ) -> EpisodicMemory:
        store = episodic_store or create_episodic_store(self.config)
        vectors = vector_store or create_vector_store(self.config)
        embeddings = embedding_provider or create_embedding_provider(self.config)
        return EpisodicMemory(
            config=self.config,
            user_id=self.user_id,
            episodic_store=store,
            vector_store=vectors,
            embedding_provider=embeddings,
        )

    def _create_semantic_memory(
        self,
        semantic_store: SemanticMemoryStore | None,
        vector_store: MilvusVectorStore | None,
        embedding_provider: EmbeddingProvider | None,
    ) -> SemanticMemory:
        store = semantic_store or create_semantic_store(self.config)
        vectors = vector_store or create_vector_store(
            self.config,
            collection_name=self.config.milvus_semantic_collection,
        )
        embeddings = embedding_provider or create_embedding_provider(self.config)
        return SemanticMemory(
            config=self.config,
            user_id=self.user_id,
            semantic_store=store,
            vector_store=vectors,
            embedding_provider=embeddings,
        )

    def add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
        auto_classify: bool = False,
    ) -> str:
        # TODO: Implement automatic memory type classification when requested.
        _ = auto_classify
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")

        return memory_module.add(
            content=content,
            importance=importance,
            metadata=metadata,
        )

    def search_memory(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[Any]:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")
        if not hasattr(memory_module, "retrieve"):
            raise ValueError(f"记忆类型 '{memory_type}' 不支持检索")
        return memory_module.retrieve(query=query, limit=limit, session_id=session_id)
