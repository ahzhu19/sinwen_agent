"""记忆模块公共类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ..config import MemoryConfig


@dataclass
class MemoryRecord:
    id: str
    content: str
    memory_type: str
    importance: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryStore:
    """临时内存 store，后续可替换为数据库实现。"""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._records_by_type: dict[str, dict[str, MemoryRecord]] = {}

    def add(self, record: MemoryRecord) -> str:
        self._records[record.id] = record
        self._records_by_type.setdefault(record.memory_type, {})[record.id] = record
        return record.id

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def list_records(self, memory_type: str | None = None) -> list[MemoryRecord]:
        if memory_type is not None:
            return list(self._records_by_type.get(memory_type, {}).values())
        return list(self._records.values())

    def remove(self, memory_id: str) -> None:
        record = self._records.pop(memory_id, None)
        if record is None:
            return

        typed_records = self._records_by_type.get(record.memory_type)
        if typed_records is None:
            return

        typed_records.pop(memory_id, None)
        if not typed_records:
            self._records_by_type.pop(record.memory_type, None)


class BaseMemory:
    memory_type: str

    def __init__(self, config: MemoryConfig, store: InMemoryStore) -> None:
        self.config = config
        self.store = store

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        memory_id = str(uuid4())
        return self.store.add(
            MemoryRecord(
                id=memory_id,
                content=content,
                memory_type=self.memory_type,
                importance=importance,
                metadata=metadata,
            )
        )
