"""消费 Neo4j SemanticOutboxEvent 并同步 Milvus 向量索引。"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig
from .storage.semantic_outbox_types import SemanticOutboxEvent


class SemanticOutboxProcessor:
    """Neo4j 事实源 + Outbox → Milvus 可重建索引。"""

    def __init__(
        self,
        config: MemoryConfig,
        semantic_store: Any,
        *,
        embedding_provider: Any | None = None,
        vector_store: Any | None = None,
    ) -> None:
        self._config = config
        self._store = semantic_store
        self._embeddings = embedding_provider
        self._vectors = vector_store

    def process_batch(self, *, batch_size: int = 20) -> tuple[int, int]:
        if not hasattr(self._store, "claim_pending_outbox_events"):
            return 0, 0

        events = self._store.claim_pending_outbox_events(batch_size=batch_size)
        succeeded = 0
        failed = 0
        for event in events:
            try:
                outcome = self._process_event(event)
            except Exception as exc:  # noqa: BLE001 — 记录后重试
                self._store.mark_outbox_failed(
                    event.event_id,
                    str(exc),
                    max_attempts=event.max_attempts,
                )
                failed += 1
                continue
            if outcome == "superseded":
                self._store.mark_outbox_superseded(event.event_id)
            else:
                self._store.mark_outbox_done(event.event_id)
            succeeded += 1
        return succeeded, failed

    def _process_event(self, event: SemanticOutboxEvent) -> str:
        state = self._store.get_memory_sync_state(event.memory_id)
        if state is None:
            return "superseded"

        if event.version < state.version:
            return "superseded"

        if event.version > state.version:
            raise ValueError(
                f"outbox 事件版本超前: event={event.version} memory={state.version}"
            )

        vector_store = self._require_vector_store(event.collection_name)

        if event.operation == "delete" or state.deleted:
            vector_store.delete(event.memory_id)
            self._store.update_embedding_sync_state(
                event.memory_id,
                embedding_version=state.version,
                embedding_status="done",
                embedding_model=event.embedding_model,
            )
            return "done"

        vector = self._require_embeddings().embed(state.content)
        vector_store.upsert(
            memory_id=event.memory_id,
            vector=vector,
            user_id=state.user_id,
            session_id=state.session_id,
        )
        self._store.update_embedding_sync_state(
            event.memory_id,
            embedding_version=state.version,
            embedding_status="done",
            embedding_model=event.embedding_model,
        )
        return "done"

    def _require_embeddings(self) -> Any:
        if self._embeddings is None:
            from .embedding import create_embedding_provider

            self._embeddings = create_embedding_provider(self._config)
        return self._embeddings

    def _require_vector_store(self, collection_name: str) -> Any:
        if self._vectors is not None and self._vectors.collection_name == collection_name:
            return self._vectors
        from .storage.milvus_store import create_vector_store

        store = create_vector_store(self._config, collection_name=collection_name)
        if self._vectors is None:
            self._vectors = store
        return store
