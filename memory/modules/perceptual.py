"""感知记忆模块。"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..config import MemoryConfig
from .base import MemoryRecord


class PerceptualMemory:
    """感知记忆：多模态元数据存储 + 按模态分离的向量检索。"""

    memory_type = "perceptual"

    def __init__(
        self,
        config: MemoryConfig,
        user_id: str,
        perceptual_store: Any,
        vector_stores: dict[str, Any],
        embedding_provider: Any,
    ) -> None:
        self.config = config
        self.user_id = user_id
        self._store = perceptual_store
        self._vectors_by_modality = vector_stores
        self._embeddings = embedding_provider

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        memory_id = str(uuid4())
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

        item = self._store.insert(
            user_id=self.user_id,
            memory_id=memory_id,
            content=content,
            modality=modality,
            importance=importance,
            raw_data=str(raw_data) if raw_data is not None else None,
            created_at=created_at,
            metadata=enriched_metadata,
        )

        vector_text = self._build_embedding_text(item.content, item.modality, item.metadata)
        vector = self._embeddings.embed(vector_text)
        session_id = item.metadata.get("session_id")
        self._vector_store_for(item.modality).upsert(
            memory_id=item.id,
            vector=vector,
            user_id=self.user_id,
            session_id=session_id if isinstance(session_id, str) else None,
        )
        return item.id

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

        items = self._store.get_many(list(hits_by_id))
        scored: list[tuple[float, MemoryRecord]] = []
        for item in items:
            vector_score, hit_modality = hits_by_id.get(item.id, (0.0, item.modality))
            recency_score = self._calculate_recency_score(item.created_at)
            importance_weight = 0.8 + (item.importance * 0.4)
            final_score = (vector_score * 0.8 + recency_score * 0.2) * importance_weight
            record = _perceptual_item_to_record(item)
            if not modality and hit_modality != "text":
                record.metadata.setdefault("cross_modal_fallback", True)
            scored.append((final_score, record))

        scored.sort(key=lambda entry: entry[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def remove(self, memory_id: str) -> None:
        item = self._store.get(memory_id)
        self._store.delete(memory_id)
        if item is None:
            return
        vector_store = self._vectors_by_modality.get(item.modality)
        if vector_store is not None:
            vector_store.delete(memory_id)

    def _calculate_recency_score(self, timestamp: str) -> float:
        try:
            memory_time = datetime.fromisoformat(timestamp)
            current_time = datetime.now(tz=memory_time.tzinfo)
            age_hours = (current_time - memory_time).total_seconds() / 3600
            decay_factor = 0.1
            recency_score = math.exp(-decay_factor * age_hours / 24)
            return max(0.1, recency_score)
        except Exception:
            return 0.5

    def _build_embedding_text(
        self,
        content: str,
        modality: str,
        metadata: dict[str, Any],
    ) -> str:
        if modality == "image":
            return str(
                metadata.get("caption")
                or metadata.get("ocr_text")
                or content
                or metadata.get("raw_data")
                or ""
            )
        if modality == "audio":
            return str(metadata.get("transcript") or content or metadata.get("raw_data") or "")
        return str(content or metadata.get("raw_data") or "")

    def _vector_store_for(self, modality: str) -> Any:
        vector_store = self._vectors_by_modality.get(modality)
        if vector_store is not None:
            return vector_store
        fallback = self._vectors_by_modality.get("text")
        if fallback is None:
            raise ValueError(f"未配置感知记忆向量集合: {modality}")
        return fallback

    def _normalize_modality(self, modality: Any) -> str:
        value = str(modality or "text").strip().lower()
        if value in {"text", "image", "audio", "video", "file"}:
            return value
        return "text"


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
