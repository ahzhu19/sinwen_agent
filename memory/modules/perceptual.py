"""感知记忆模块。"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..config import MemoryConfig
from ..storage.vector_outbox import VectorWriteError
from ..vector_outbox_processor import VectorOutboxProcessor
from .base import MemoryRecord

_ALLOWED_MODALITIES = frozenset({"text", "image", "audio", "video", "file"})


class PerceptualMemory:
    """感知记忆（experimental）：多模态元数据 + 按模态 Milvus 向量。

    当前限制：元数据仅进程内存；图像/音频使用 caption/transcript 文本代理 embedding，
    非 CLIP/CLAP。默认 ``enable_perceptual=False``，生产环境请优先 RAG（文档）或 semantic（事实）。
    """

    memory_type = "perceptual"

    def __init__(
        self,
        config: MemoryConfig,
        user_id: str,
        perceptual_store: Any,
        vector_stores: dict[str, Any],
        embedding_provider: Any,
        pg_vector_outbox: Any | None = None,
        outbox_processor: VectorOutboxProcessor | None = None,
    ) -> None:
        self.config = config
        self.user_id = user_id
        self._store = perceptual_store
        self._vectors_by_modality = vector_stores
        self._embeddings = embedding_provider
        self._pg_vector_outbox = pg_vector_outbox
        self._outbox_processor = outbox_processor

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        memory_id = str(uuid4())
        item = self._insert_metadata(
            memory_id=memory_id,
            content=content,
            importance=importance,
            metadata=metadata,
        )
        self._write_vector(item)
        return item.id

    def update(
        self,
        memory_id: str,
        *,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        existing = self._store.get(memory_id)
        if existing is None or existing.user_id != self.user_id:
            raise KeyError(f"未找到记忆: {memory_id}")

        enriched_metadata = dict(metadata)
        modality = self._normalize_modality(
            enriched_metadata.get("modality", existing.modality)
        )
        raw_data = enriched_metadata.get("raw_data", existing.raw_data)
        created_at = existing.created_at
        enriched_metadata.setdefault("modality", modality)
        enriched_metadata.setdefault("timestamp", created_at)
        if raw_data is not None:
            enriched_metadata.setdefault("raw_data", str(raw_data))

        old_modality = existing.modality
        item = self._store.update(
            memory_id=memory_id,
            user_id=self.user_id,
            content=content,
            modality=modality,
            importance=importance,
            raw_data=str(raw_data) if raw_data is not None else None,
            created_at=created_at,
            metadata=enriched_metadata,
        )

        if modality != old_modality:
            self._delete_vector(
                memory_id,
                old_modality,
                existing.metadata.get("session_id"),
            )
        self._write_vector(item)
        return memory_id

    def _insert_metadata(
        self,
        *,
        memory_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> Any:
        enriched_metadata = dict(metadata)
        modality = self._normalize_modality(enriched_metadata.get("modality"))
        raw_data = enriched_metadata.get("raw_data")
        created_at = str(
            enriched_metadata.get("timestamp")
            or enriched_metadata.get("created_at")
            or datetime.now().isoformat()
        )
        enriched_metadata.setdefault("modality", modality)
        enriched_metadata.setdefault("timestamp", created_at)
        if raw_data is not None:
            enriched_metadata.setdefault("raw_data", str(raw_data))

        return self._store.insert(
            user_id=self.user_id,
            memory_id=memory_id,
            content=content,
            modality=modality,
            importance=importance,
            raw_data=str(raw_data) if raw_data is not None else None,
            created_at=created_at,
            metadata=enriched_metadata,
        )

    def _write_vector(self, item: Any) -> None:
        vector_text = self._build_embedding_text(item.content, item.modality, item.metadata)
        vector = self._embeddings.embed(vector_text)
        session_id = item.metadata.get("session_id")
        session_value = session_id if isinstance(session_id, str) else None
        vector_store = self._vector_store_for(item.modality)

        if self._pg_vector_outbox is not None and self.config.enable_persistent_vector_outbox:
            self._pg_vector_outbox.enqueue_upsert(
                memory_kind="perceptual",
                memory_id=item.id,
                user_id=self.user_id,
                session_id=session_value,
                collection_name=vector_store.collection_name,
                vector=vector,
                max_attempts=self.config.vector_outbox_max_attempts,
            )
            return

        try:
            vector_store.upsert(
                memory_id=item.id,
                vector=vector,
                user_id=self.user_id,
                session_id=session_value,
            )
        except Exception as exc:
            raise VectorWriteError(
                f"感知记忆 {item.id} 元数据已写入，但 Milvus 向量写入失败: {exc}"
            ) from exc

    def _delete_vector(
        self,
        memory_id: str,
        modality: str,
        session_id: Any,
    ) -> None:
        if self._pg_vector_outbox is not None:
            vector_store = self._vector_store_for(modality)
            session_value = session_id if isinstance(session_id, str) else None
            self._pg_vector_outbox.enqueue_delete(
                memory_kind="perceptual",
                memory_id=memory_id,
                user_id=self.user_id,
                session_id=session_value,
                collection_name=vector_store.collection_name,
                max_attempts=self.config.vector_outbox_max_attempts,
            )
        try:
            self._vector_store_for(modality).delete(memory_id)
        except Exception:
            if self._pg_vector_outbox is None:
                raise

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        session_id: str | None = None,
        modality: str | None = None,
        **kwargs: Any,
    ) -> list[MemoryRecord]:
        _ = kwargs
        query_vector = self._embeddings.embed(query)
        modalities = [self._normalize_modality(modality)] if modality else list(
            self._vectors_by_modality
        )

        hits_by_id: dict[str, tuple[float, str]] = {}
        for candidate_modality in modalities:
            vector_store = self._vectors_by_modality.get(candidate_modality)
            if vector_store is None:
                continue
            hits = vector_store.search(
                query_vector=query_vector,
                user_id=self.user_id,
                limit=limit,
                session_id=session_id,
            )
            for hit in hits:
                current = hits_by_id.get(hit.memory_id)
                if current is None or hit.score > current[0]:
                    hits_by_id[hit.memory_id] = (hit.score, candidate_modality)

        if not hits_by_id:
            return []

        items = self._store.get_many(list(hits_by_id.keys()))
        scored: list[tuple[float, MemoryRecord]] = []
        for item in items:
            vector_score, hit_modality = hits_by_id.get(item.id, (0.0, item.modality))
            recency_score = self._calculate_recency_score(item.created_at)
            importance_weight = 0.8 + (item.importance * 0.4)
            final_score = (vector_score * 0.8 + recency_score * 0.2) * importance_weight
            scored.append((final_score, _perceptual_item_to_record(item)))

        scored.sort(key=lambda entry: entry[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def remove(self, memory_id: str) -> None:
        item = self._store.get(memory_id)
        if item is not None and self._pg_vector_outbox is not None:
            vector_store = self._vector_store_for(item.modality)
            self._pg_vector_outbox.enqueue_delete(
                memory_kind="perceptual",
                memory_id=memory_id,
                user_id=self.user_id,
                session_id=item.metadata.get("session_id")
                if isinstance(item.metadata.get("session_id"), str)
                else None,
                collection_name=vector_store.collection_name,
                max_attempts=self.config.vector_outbox_max_attempts,
            )
        self._store.delete(memory_id)
        if item is None:
            return
        try:
            self._vectors_by_modality[item.modality].delete(memory_id)
        except Exception:
            if self._pg_vector_outbox is None:
                raise

    def _calculate_recency_score(self, timestamp: str) -> float:
        try:
            memory_time = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise ValueError(f"感知记忆 timestamp 必须是 ISO-8601 格式: {timestamp!r}") from exc
        current_time = datetime.now(tz=memory_time.tzinfo)
        age_hours = (current_time - memory_time).total_seconds() / 3600
        decay_factor = 0.1
        recency_score = math.exp(-decay_factor * age_hours / 24)
        return max(0.1, recency_score)

    def _build_embedding_text(
        self,
        content: str,
        modality: str,
        metadata: dict[str, Any],
    ) -> str:
        if modality == "image":
            proxy = (
                metadata.get("caption")
                or metadata.get("ocr_text")
                or content
                or metadata.get("raw_data")
            )
            if not proxy:
                raise ValueError("图像感知记忆需要 caption、ocr_text 或 content 作为文本代理")
            return str(proxy)
        if modality == "audio":
            proxy = metadata.get("transcript") or content or metadata.get("raw_data")
            if not proxy:
                raise ValueError("音频感知记忆需要 transcript 或 content 作为文本代理")
            return str(proxy)
        text = content or metadata.get("raw_data")
        if not text:
            raise ValueError("感知记忆需要 content 或 raw_data")
        return str(text)

    def _vector_store_for(self, modality: str) -> Any:
        vector_store = self._vectors_by_modality.get(modality)
        if vector_store is None:
            raise ValueError(f"未配置感知记忆向量集合: {modality}")
        return vector_store

    def _normalize_modality(self, modality: Any) -> str:
        if modality is None or str(modality).strip() == "":
            return "text"
        value = str(modality).strip().lower()
        if value not in _ALLOWED_MODALITIES:
            raise ValueError(
                f"不支持的感知模态 '{modality}'，允许: {', '.join(sorted(_ALLOWED_MODALITIES))}"
            )
        return value


def _perceptual_item_to_record(item: Any) -> MemoryRecord:
    metadata = dict(item.metadata)
    metadata.setdefault("modality", item.modality)
    metadata.setdefault("raw_data", item.raw_data)
    metadata.setdefault("timestamp", item.created_at)
    return MemoryRecord(
        id=item.id,
        content=item.content,
        memory_type="perceptual",
        importance=item.importance,
        metadata=metadata,
    )
