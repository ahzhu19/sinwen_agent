"""MemoryRecord 与各存储实体之间的转换。"""

from __future__ import annotations

from typing import Any

from .modules.base import MemoryRecord


def episodic_event_to_record(event: Any) -> MemoryRecord:
    meta = dict(event.metadata)
    meta.setdefault("session_id", event.session_id)
    meta.setdefault("occurred_at", event.occurred_at.timestamp())
    meta.setdefault("created_at", event.created_at.timestamp())
    meta.setdefault("sequence_no", event.sequence_no)
    return MemoryRecord(
        id=event.id,
        content=event.content,
        memory_type="episodic",
        importance=event.importance,
        metadata=meta,
    )


def semantic_fact_to_record(fact: Any) -> MemoryRecord:
    metadata = dict(fact.metadata)
    metadata.setdefault("concepts", list(fact.concepts))
    return MemoryRecord(
        id=fact.id,
        content=fact.content,
        memory_type="semantic",
        importance=fact.importance,
        metadata=metadata,
    )


def perceptual_item_to_record(item: Any) -> MemoryRecord:
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
