"""RAG vector outbox processor."""

from __future__ import annotations

from typing import Any

from .config import RagConfig
from .outbox_store import RagVectorOutboxRecord, RagVectorOutboxStore
from .storage import RagStore


class RagVectorOutboxProcessor:
    def __init__(
        self,
        config: RagConfig,
        outbox_store: RagVectorOutboxStore,
        *,
        vector_store: Any,
        rag_store: RagStore,
    ) -> None:
        self._config = config
        self._outbox = outbox_store
        self._vectors = vector_store
        self._rag_store = rag_store

    def process_batch(self, *, batch_size: int = 20, reclaim_stale: bool = True) -> tuple[int, int]:
        if reclaim_stale and hasattr(self._outbox, "reclaim_stale_processing"):
            self._outbox.reclaim_stale_processing(
                timeout_seconds=self._config.rag_vector_outbox_processing_timeout_seconds,
            )

        entries = self._outbox.claim_pending(batch_size=batch_size)
        succeeded = 0
        failed = 0
        for entry in entries:
            try:
                self._process_entry(entry)
            except Exception as exc:  # noqa: BLE001
                self._outbox.mark_failed(entry.id, str(exc), max_attempts=entry.max_attempts)
                failed += 1
                continue
            self._outbox.mark_done(entry.id)
            succeeded += 1
        return succeeded, failed

    def _process_entry(self, entry: RagVectorOutboxRecord) -> None:
        chunks = self._rag_store.get_chunks([entry.chunk_id])
        if not chunks:
            raise ValueError(f"RAG chunk 不存在: {entry.chunk_id}")
        document = self._rag_store.get_document(entry.document_id)
        chunk = chunks[0]
        self._vectors.upsert_many([(chunk, entry.vector, document)])
        self._rag_store.mark_chunks_indexed([entry.chunk_id])
