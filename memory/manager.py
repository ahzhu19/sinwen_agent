"""记忆管理器：承载核心记忆管理逻辑。"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig
from .embedding import EmbeddingProvider, create_embedding_provider
from .modules import (
    EpisodicMemory,
    InMemoryStore,
    PerceptualMemory,
    SemanticMemory,
    WorkingMemory,
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
from .forget_policy import (
    default_forget_limit,
    default_forget_threshold,
    parse_occurred_at,
    should_forget_record,
)
from .semantic_outbox_processor import SemanticOutboxProcessor
from .storage.vector_outbox import VectorOutbox
from .vector_outbox_processor import VectorOutboxProcessor

_VECTOR_MEMORY_TYPES = frozenset({"episodic", "semantic", "perceptual"})


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
        self.store = InMemoryStore()
        self.memory_modules: dict[str, Any] = {}
        self.pg_vector_outbox: PostgresVectorOutboxStore | None = None
        self.vector_outbox: VectorOutbox | None = None
        self._outbox_processor: VectorOutboxProcessor | None = None
        self._concept_extractor = concept_extractor

        if (
            config.enable_persistent_vector_outbox
            and config.enable_vector_outbox
            and config.database_url
        ):
            self.pg_vector_outbox = create_postgres_outbox_store(config)
            self._outbox_processor = VectorOutboxProcessor(
                config,
                self.pg_vector_outbox,
            )
        elif config.enable_vector_outbox:
            self.vector_outbox = VectorOutbox(
                max_attempts=config.vector_outbox_max_attempts,
            )

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
            self.memory_modules["perceptual"] = self._create_perceptual_memory(
                perceptual_store=perceptual_store,
                vector_stores=perceptual_vector_stores,
                embedding_provider=perceptual_embedding_provider,
            )

        if self._outbox_processor is not None and "episodic" in self.memory_modules:
            episodic_module = self.memory_modules["episodic"]
            store = getattr(episodic_module, "_store", None)
            if store is not None and hasattr(store, "mark_vector_indexed"):
                self._outbox_processor._episodic_store = store

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
            vector_outbox=self.vector_outbox,
            pg_vector_outbox=self.pg_vector_outbox,
            outbox_processor=self._outbox_processor,
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
            collection_name=self.config.semantic_milvus_collection(),
        )
        embeddings = embedding_provider or create_embedding_provider(self.config)
        if not hasattr(store, "claim_pending_outbox_events"):
            raise ValueError("语义记忆 store 需支持 Neo4j Transactional Outbox")
        semantic_outbox_processor = SemanticOutboxProcessor(
            self.config,
            store,
            embedding_provider=embeddings,
            vector_store=vectors,
        )
        return SemanticMemory(
            config=self.config,
            user_id=self.user_id,
            semantic_store=store,
            vector_store=vectors,
            embedding_provider=embeddings,
            concept_extractor=self._concept_extractor,
            semantic_outbox_processor=semantic_outbox_processor,
        )

    def _create_perceptual_memory(
        self,
        perceptual_store: PerceptualMemoryStore | None,
        vector_stores: dict[str, MilvusVectorStore] | None,
        embedding_provider: EmbeddingProvider | None,
    ) -> PerceptualMemory:
        store = perceptual_store or create_perceptual_store()
        vectors = vector_stores or {
            "text": create_vector_store(
                self.config,
                collection_name=self.config.perceptual_milvus_collection("text"),
            ),
            "image": create_vector_store(
                self.config,
                collection_name=self.config.perceptual_milvus_collection("image"),
            ),
            "audio": create_vector_store(
                self.config,
                collection_name=self.config.perceptual_milvus_collection("audio"),
            ),
            "video": create_vector_store(
                self.config,
                collection_name=self.config.perceptual_milvus_collection("video"),
            ),
            "file": create_vector_store(
                self.config,
                collection_name=self.config.perceptual_milvus_collection("file"),
            ),
        }
        embeddings = embedding_provider or create_embedding_provider(self.config)
        return PerceptualMemory(
            config=self.config,
            user_id=self.user_id,
            perceptual_store=store,
            vector_stores=vectors,
            embedding_provider=embeddings,
            pg_vector_outbox=self.pg_vector_outbox,
            outbox_processor=self._outbox_processor,
        )

    def add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")

        return memory_module.add(
            content=content,
            importance=importance,
            metadata=metadata,
        )

    def flush_vector_outbox(self) -> dict[str, tuple[int, int]]:
        """补偿 Milvus 双写失败条目。返回各类型 (成功数, 仍失败数)。"""
        batch_size = self.config.vector_outbox_worker_batch_size
        results: dict[str, tuple[int, int]] = {}

        if self._outbox_processor is not None:
            for kind in ("episodic", "perceptual"):
                if kind in self.memory_modules:
                    results[kind] = self._outbox_processor.process_batch(
                        batch_size=batch_size,
                        memory_kind=kind,
                    )

        if "semantic" in self.memory_modules and hasattr(
            self.memory_modules["semantic"],
            "flush_vector_outbox",
        ):
            results["semantic"] = self.memory_modules["semantic"].flush_vector_outbox()

        if results:
            return results

        if "episodic" in self.memory_modules and hasattr(
            self.memory_modules["episodic"],
            "flush_vector_outbox",
        ):
            results["episodic"] = self.memory_modules["episodic"].flush_vector_outbox()
        return results

    def vector_outbox_pending(self) -> int:
        pending = 0
        if self.pg_vector_outbox is not None:
            pending += self.pg_vector_outbox.pending_count()
        semantic = self.memory_modules.get("semantic")
        if semantic is not None:
            store = getattr(semantic, "_store", None)
            if store is not None and hasattr(store, "pending_outbox_count"):
                pending += store.pending_outbox_count()
        if pending:
            return pending
        if self.vector_outbox is None:
            return 0
        return self.vector_outbox.pending_count()

    def search_memory(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[Any]:
        if memory_type in _VECTOR_MEMORY_TYPES and self.config.vector_outbox_poll_on_read:
            self.flush_vector_outbox()
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")
        if not hasattr(memory_module, "retrieve"):
            raise ValueError(f"记忆类型 '{memory_type}' 不支持检索")
        return memory_module.retrieve(query=query, limit=limit, session_id=session_id)

    def remove_memory(self, memory_id: str, memory_type: str) -> None:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")
        if not hasattr(memory_module, "remove"):
            raise ValueError(f"记忆类型 '{memory_type}' 不支持删除")
        memory_module.remove(memory_id)

    def update_memory(
        self,
        memory_id: str,
        memory_type: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")

        if memory_type == "working":
            if not hasattr(memory_module, "update"):
                raise ValueError(f"记忆类型 '{memory_type}' 不支持更新")
            updated = memory_module.update(
                memory_id,
                content=content,
                importance=importance,
                metadata=metadata,
            )
            if updated is None:
                raise KeyError(f"未找到记忆: {memory_id}")
            return updated.id

        existing = self._get_record(memory_type, memory_id)
        if existing is None:
            raise KeyError(f"未找到记忆: {memory_id}")

        merged_metadata = dict(existing.metadata)
        if metadata:
            merged_metadata.update(metadata)
        new_content = content if content is not None else existing.content
        new_importance = importance if importance is not None else existing.importance

        if not hasattr(memory_module, "update"):
            raise ValueError(f"记忆类型 '{memory_type}' 不支持更新")
        return memory_module.update(
            memory_id,
            content=new_content,
            importance=new_importance,
            metadata=merged_metadata,
        )

    def memory_stats(self, session_id: str | None = None) -> dict[str, Any]:
        stats: dict[str, Any] = {"user_id": self.user_id, "enabled_types": list(self.memory_modules)}
        counts: dict[str, int] = {}

        if "working" in self.memory_modules:
            records = self.memory_modules["working"].store.list_records(memory_type="working")
            if session_id:
                records = [r for r in records if r.metadata.get("session_id") == session_id]
            counts["working"] = len(records)

        if "episodic" in self.memory_modules:
            timeline = self.memory_modules["episodic"].list_timeline(
                session_id=session_id,
                limit=10_000,
            )
            counts["episodic"] = len(timeline)

        if "perceptual" in self.memory_modules:
            store = self.memory_modules["perceptual"]._store
            items = [
                item
                for item in getattr(store, "items", {}).values()
                if getattr(item, "user_id", None) == self.user_id
            ]
            if session_id:
                items = [item for item in items if item.metadata.get("session_id") == session_id]
            counts["perceptual"] = len(items)

        if "semantic" in self.memory_modules:
            counts["semantic"] = self.memory_modules["semantic"].count_for_user(session_id)

        stats["counts"] = counts
        return stats

    def memory_summary(
        self,
        session_id: str | None = None,
        limit_per_type: int = 3,
    ) -> dict[str, list[Any]]:
        summary: dict[str, list[Any]] = {}
        if "working" in self.memory_modules and session_id:
            summary["working"] = self.memory_modules["working"].list_recent(session_id)[
                -limit_per_type:
            ]
        if "episodic" in self.memory_modules:
            summary["episodic"] = self.memory_modules["episodic"].list_timeline(
                session_id=session_id,
                limit=limit_per_type,
            )
        if "semantic" in self.memory_modules:
            summary["semantic"] = self.memory_modules["semantic"].list_for_user(
                session_id=session_id,
                limit=limit_per_type,
            )
        return summary

    def forget_memories(
        self,
        memory_type: str = "working",
        *,
        strategy: str = "importance",
        session_id: str | None = None,
        importance_threshold: float | None = None,
        older_than_days: int | None = None,
        limit: int | None = None,
    ) -> int:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")

        threshold = (
            importance_threshold
            if importance_threshold is not None
            else default_forget_threshold(memory_type)
        )
        max_remove = limit if limit is not None else default_forget_limit(memory_type)

        if memory_type == "working":
            memory_module.cleanup_expired()
            removed = 0
            for record in list(memory_module.store.list_records(memory_type="working")):
                if removed >= max_remove:
                    break
                if session_id and record.metadata.get("session_id") != session_id:
                    continue
                if should_forget_record(
                    importance=record.importance,
                    importance_threshold=threshold,
                    strategy=strategy,
                ):
                    memory_module.remove(record.id)
                    removed += 1
            return removed

        if memory_type == "episodic":
            store = memory_module._store
            if not hasattr(store, "list_for_forget"):
                raise ValueError("episodic store 需实现 list_for_forget")
            removed = 0
            for event in store.list_for_forget(self.user_id, session_id=session_id, limit=10_000):
                if removed >= max_remove:
                    break
                if should_forget_record(
                    importance=event.importance,
                    importance_threshold=threshold,
                    strategy=strategy,
                    occurred_at=event.occurred_at,
                    older_than_days=older_than_days,
                ):
                    memory_module.remove(event.id)
                    removed += 1
            return removed

        if memory_type == "semantic":
            removed = 0
            facts = memory_module.list_for_user(session_id=session_id, limit=10_000)
            for record in facts:
                if removed >= max_remove:
                    break
                if should_forget_record(
                    importance=record.importance,
                    importance_threshold=threshold,
                    strategy=strategy,
                    occurred_at=parse_occurred_at(record),
                    older_than_days=older_than_days,
                ):
                    memory_module.remove(record.id)
                    removed += 1
            return removed

        if memory_type == "perceptual":
            store = memory_module._store
            if not hasattr(store, "list_by_user"):
                raise ValueError("perceptual store 需实现 list_by_user")
            removed = 0
            for item in store.list_by_user(self.user_id, session_id=session_id, limit=10_000):
                if removed >= max_remove:
                    break
                if should_forget_record(
                    importance=item.importance,
                    importance_threshold=threshold,
                    strategy=strategy,
                    occurred_at=parse_occurred_at(item),
                    older_than_days=older_than_days,
                ):
                    memory_module.remove(item.id)
                    removed += 1
            return removed

        raise ValueError(f"记忆类型 '{memory_type}' 暂不支持 forget 策略")

    def consolidate_working_to_episodic(
        self,
        session_id: str,
        *,
        importance_threshold: float = 0.5,
    ) -> list[str]:
        if "working" not in self.memory_modules:
            raise ValueError("未启用 working 记忆")
        if "episodic" not in self.memory_modules:
            raise ValueError("未启用 episodic 记忆，无法整合")

        working = self.memory_modules["working"]
        episodic = self.memory_modules["episodic"]
        created_ids: list[str] = []
        for record in working.list_recent(session_id):
            if record.importance < importance_threshold:
                continue
            memory_id = episodic.add(
                content=record.content,
                importance=record.importance,
                metadata={
                    **dict(record.metadata),
                    "consolidated_from": record.id,
                    "session_id": session_id,
                },
            )
            working.remove(record.id)
            created_ids.append(memory_id)
        return created_ids

    def clear_memories(
        self,
        memory_type: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, int]:
        targets = [memory_type] if memory_type else list(self.memory_modules)
        cleared: dict[str, int] = {}

        for target in targets:
            module = self.memory_modules.get(target)
            if module is None:
                continue

            if target == "working":
                if session_id:
                    before = len(module.store.list_records(memory_type="working"))
                    module.clear_session(session_id)
                    after = len(module.store.list_records(memory_type="working"))
                    cleared[target] = before - after
                else:
                    records = list(module.store.list_records(memory_type="working"))
                    for record in records:
                        module.remove(record.id)
                    cleared[target] = len(records)
                continue

            if target == "episodic":
                timeline = module.list_timeline(session_id=session_id, limit=10_000)
                for record in timeline:
                    module.remove(record.id)
                cleared[target] = len(timeline)
                continue

            if target == "perceptual":
                store = module._store
                items = list(getattr(store, "items", {}).values())
                count = 0
                for item in items:
                    if item.user_id != self.user_id:
                        continue
                    if session_id and item.metadata.get("session_id") != session_id:
                        continue
                    module.remove(item.id)
                    count += 1
                cleared[target] = count
                continue

            if target == "semantic":
                cleared[target] = self.memory_modules["semantic"].remove_all_for_user(
                    session_id
                )
                continue

        return cleared

    def _get_record(self, memory_type: str, memory_id: str) -> Any | None:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            return None
        if hasattr(memory_module, "get"):
            return memory_module.get(memory_id)
        return None
