"""PostgreSQL 情景记忆结构化存储。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import MemoryConfig


@dataclass(frozen=True)
class EpisodicEvent:
    id: str
    user_id: str
    session_id: str | None
    content: str
    importance: float
    occurred_at: datetime
    created_at: datetime
    sequence_no: int
    metadata: dict[str, Any]


class EpisodicMemoryStore(Protocol):
    def ensure_schema(self) -> None:
        ...

    def insert(
        self,
        user_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        session_id: str | None = None,
        occurred_at: datetime | None = None,
        memory_id: str | None = None,
    ) -> EpisodicEvent:
        ...

    def get(self, memory_id: str) -> EpisodicEvent | None:
        ...

    def get_many(self, memory_ids: list[str]) -> list[EpisodicEvent]:
        ...

    def list_timeline(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[EpisodicEvent]:
        ...

    def delete(self, memory_id: str) -> None:
        ...


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodic_memories (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT,
    content TEXT NOT NULL,
    importance DOUBLE PRECISION NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    sequence_no BIGSERIAL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_episodic_user_session_seq
    ON episodic_memories (user_id, session_id, sequence_no);
"""


class PostgresEpisodicMemoryStore:
    """PostgreSQL 情景记忆存储实现。"""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with psycopg.connect(self._database_url) as conn:
            conn.execute(_SCHEMA_SQL)
            conn.commit()
        self._schema_ready = True

    def insert(
        self,
        user_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        session_id: str | None = None,
        occurred_at: datetime | None = None,
        memory_id: str | None = None,
    ) -> EpisodicEvent:
        self.ensure_schema()
        event_id = memory_id or str(uuid4())
        occurred = occurred_at or datetime.now(timezone.utc)
        meta = dict(metadata)
        meta.setdefault("session_id", session_id)

        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                INSERT INTO episodic_memories (
                    id, user_id, session_id, content, importance, occurred_at, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id, user_id, session_id, content, importance,
                          occurred_at, created_at, sequence_no, metadata
                """,
                (
                    event_id,
                    user_id,
                    session_id,
                    content,
                    importance,
                    occurred,
                    Jsonb(meta),
                ),
            ).fetchone()
            conn.commit()

        assert row is not None
        return _row_to_event(row)

    def get(self, memory_id: str) -> EpisodicEvent | None:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                SELECT id, user_id, session_id, content, importance,
                       occurred_at, created_at, sequence_no, metadata
                FROM episodic_memories
                WHERE id = %s
                """,
                (memory_id,),
            ).fetchone()
        return _row_to_event(row) if row else None

    def get_many(self, memory_ids: list[str]) -> list[EpisodicEvent]:
        if not memory_ids:
            return []
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, session_id, content, importance,
                       occurred_at, created_at, sequence_no, metadata
                FROM episodic_memories
                WHERE id = ANY(%s)
                """,
                (memory_ids,),
            ).fetchall()
        events = [_row_to_event(row) for row in rows]
        order = {memory_id: index for index, memory_id in enumerate(memory_ids)}
        events.sort(key=lambda event: order.get(event.id, len(memory_ids)))
        return events

    def list_timeline(
        self,
        user_id: str,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[EpisodicEvent]:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            if session_id is None:
                rows = conn.execute(
                    """
                    SELECT id, user_id, session_id, content, importance,
                           occurred_at, created_at, sequence_no, metadata
                    FROM episodic_memories
                    WHERE user_id = %s
                    ORDER BY sequence_no DESC
                    LIMIT %s
                    """,
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, user_id, session_id, content, importance,
                           occurred_at, created_at, sequence_no, metadata
                    FROM episodic_memories
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY sequence_no DESC
                    LIMIT %s
                    """,
                    (user_id, session_id, limit),
                ).fetchall()
        events = [_row_to_event(row) for row in rows]
        events.reverse()
        return events

    def delete(self, memory_id: str) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            conn.execute("DELETE FROM episodic_memories WHERE id = %s", (memory_id,))
            conn.commit()


def create_episodic_store(config: MemoryConfig) -> PostgresEpisodicMemoryStore:
    if not config.database_url:
        raise ValueError("未配置 DATABASE_URL，无法连接 PostgreSQL")
    return PostgresEpisodicMemoryStore(config.database_url)


def _row_to_event(row: dict[str, Any]) -> EpisodicEvent:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return EpisodicEvent(
        id=str(row["id"]),
        user_id=row["user_id"],
        session_id=row["session_id"],
        content=row["content"],
        importance=float(row["importance"]),
        occurred_at=row["occurred_at"],
        created_at=row["created_at"],
        sequence_no=int(row["sequence_no"]),
        metadata=dict(metadata or {}),
    )
