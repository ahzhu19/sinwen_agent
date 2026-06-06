"""处理 Postgres memory_vector_outbox 并写入 Milvus。"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig
from .storage.milvus_store import create_vector_store
from .storage.postgres_outbox_store import PostgresVectorOutboxStore, VectorOutboxRecord


class VectorOutboxProcessor:
    """将 outbox 条目同步到 Milvus。"""

    def __init__(
        self,
        config: MemoryConfig,
        outbox_store: PostgresVectorOutboxStore,
        *,
        vector_stores_by_collection: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._outbox = outbox_store
        self._vector_stores = vector_stores_by_collection or _default_vector_stores(config)

    def process_batch(
        self,
        *,
        batch_size: int = 20,
        memory_kind: str | None = None,
    ) -> tuple[int, int]:
        """返回 (成功数, 失败数)。"""
        entries = self._outbox.claim_pending(batch_size=batch_size)
        if memory_kind is not None:
            entries = [entry for entry in entries if entry.memory_kind == memory_kind]

        succeeded = 0
        failed = 0
        for entry in entries:
            try:
                self._process_entry(entry)
            except Exception as exc:  # noqa: BLE001 — 记录后重试
                self._outbox.mark_failed(
                    entry.id,
                    str(exc),
                    max_attempts=entry.max_attempts,
                )
                failed += 1
                continue
            self._outbox.mark_done(entry.id)
            succeeded += 1
        return succeeded, failed

    def _process_entry(self, entry: VectorOutboxRecord) -> None:
        vector_store = self._vector_stores.get(entry.collection_name)
        if vector_store is None:
            raise ValueError(f"未配置 collection 对应的向量存储: {entry.collection_name}")

        if entry.op == "delete":
            vector_store.delete(entry.memory_id)
            return

        if entry.vector is None:
            raise ValueError(f"outbox upsert 缺少 vector: {entry.memory_id}")
        vector_store.upsert(
            memory_id=entry.memory_id,
            vector=entry.vector,
            user_id=entry.user_id,
            session_id=entry.session_id,
        )


def _default_vector_stores(config: MemoryConfig) -> dict[str, Any]:
    episodic = create_vector_store(config)
    semantic = create_vector_store(config, collection_name=config.milvus_semantic_collection)
    perceptual_text = create_vector_store(
        config,
        collection_name=config.milvus_perceptual_text_collection,
    )
    perceptual_image = create_vector_store(
        config,
        collection_name=config.milvus_perceptual_image_collection,
    )
    perceptual_audio = create_vector_store(
        config,
        collection_name=config.milvus_perceptual_audio_collection,
    )
    perceptual_video = create_vector_store(
        config,
        collection_name=config.milvus_perceptual_video_collection,
    )
    perceptual_file = create_vector_store(
        config,
        collection_name=config.milvus_perceptual_file_collection,
    )
    return {
        episodic.collection_name: episodic,
        semantic.collection_name: semantic,
        perceptual_text.collection_name: perceptual_text,
        perceptual_image.collection_name: perceptual_image,
        perceptual_audio.collection_name: perceptual_audio,
        perceptual_video.collection_name: perceptual_video,
        perceptual_file.collection_name: perceptual_file,
    }
