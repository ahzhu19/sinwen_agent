"""文档/感知记忆元数据存储。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class PerceptualItem:
    id: str
    user_id: str
    content: str
    modality: str
    importance: float
    raw_data: str | None
    created_at: str
    metadata: dict[str, Any]


class PerceptualMemoryStore(Protocol):
    def insert(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        modality: str,
        importance: float,
        raw_data: str | None,
        created_at: str,
        metadata: dict[str, Any],
    ) -> PerceptualItem:
        ...

    def get(self, memory_id: str) -> PerceptualItem | None:
        ...

    def get_many(self, memory_ids: list[str]) -> list[PerceptualItem]:
        ...

    def delete(self, memory_id: str) -> None:
        ...


class InMemoryPerceptualStore:
    """第一版感知记忆元数据 store，用于本地开发和测试。"""

    def __init__(self) -> None:
        self.items: dict[str, PerceptualItem] = {}

    def insert(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        modality: str,
        importance: float,
        raw_data: str | None,
        created_at: str,
        metadata: dict[str, Any],
    ) -> PerceptualItem:
        item = PerceptualItem(
            id=memory_id,
            user_id=user_id,
            content=content,
            modality=modality,
            importance=importance,
            raw_data=raw_data,
            created_at=created_at,
            metadata=dict(metadata),
        )
        self.items[memory_id] = item
        return item

    def get(self, memory_id: str) -> PerceptualItem | None:
        return self.items.get(memory_id)

    def get_many(self, memory_ids: list[str]) -> list[PerceptualItem]:
        return [self.items[memory_id] for memory_id in memory_ids if memory_id in self.items]

    def delete(self, memory_id: str) -> None:
        self.items.pop(memory_id, None)


def create_perceptual_store() -> InMemoryPerceptualStore:
    return InMemoryPerceptualStore()
