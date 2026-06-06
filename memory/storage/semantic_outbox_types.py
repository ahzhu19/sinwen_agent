"""Neo4j 语义记忆 outbox 共享类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SemanticOutboxEvent:
    event_id: str
    memory_id: str
    user_id: str
    version: int
    operation: str
    status: str
    attempts: int
    max_attempts: int
    last_error: str | None
    embedding_model: str
    collection_name: str
    session_id: str | None = None


@dataclass(frozen=True)
class SemanticMemorySyncState:
    id: str
    user_id: str
    content: str
    importance: float
    concepts: list[str]
    metadata: dict[str, Any]
    version: int
    embedding_version: int
    embedding_status: str
    embedding_model: str
    deleted: bool
    session_id: str | None = None
