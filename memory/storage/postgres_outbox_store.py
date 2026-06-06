"""PostgreSQL 持久化 Milvus 向量 outbox。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..config import MemoryConfig

VectorOutboxKind = Literal["episodic", "perceptual"]
VectorOutboxOp = Literal["upsert", "delete"]
VectorOutboxStatus = Literal["pending", "processing", "done", "dead"]


@dataclass(frozen=True)
class VectorOutboxRecord:
    id: int
    memory_kind: str
    memory_id: str
    user_id: str
    session_id: str | None
    collection_name: str
    op: str
    vector: list[float] | None
    status: str
    attempts: int
    max_attempts: int


_OUTBOX_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_vector_outbox (
    id BIGSERIAL PRIMARY KEY,
    memory_kind TEXT NOT NULL,
    memory_id UUID NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT,
    collection_name TEXT NOT NULL,
    op TEXT NOT NULL DEFAULT 'upsert',
    vector JSONB,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 5,
    last_error TEXT,
    next_retry_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (memory_kind, memory_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_vector_outbox_pending
    ON memory_vector_outbox (status, next_retry_at)
    WHERE status IN ('pending', 'processing');
"""


class PostgresVectorOutboxStore(Protocol):
    def ensure_schema(self) -> None:
        ...

    def pending_count(self, *, memory_kind: str | None = None) -> int:
        ...

    def status_counts(self) -> dict[str, dict[str, int]]:
        ...

    def enqueue_upsert(
        self,
        *,
        memory_kind: VectorOutboxKind,
        memory_id: str,
        user_id: str,
        session_id: str | None,
        collection_name: str,
        vector: list[float],
        max_attempts: int,
    ) -> None:
        ...

    def enqueue_delete(
        self,
        *,
        memory_kind: VectorOutboxKind,
        memory_id: str,
        user_id: str,
        session_id: str | None,
        collection_name: str,
        max_attempts: int,
    ) -> None:
        ...

    def claim_pending(self, *, batch_size: int = 20) -> list[VectorOutboxRecord]:
        ...

    def mark_done(self, entry_id: int) -> None:
        ...

    def mark_failed(self, entry_id: int, error: str, *, max_attempts: int) -> None:
        ...

    def reclaim_stale_processing(self, *, timeout_seconds: int) -> int:
        ...

    def replay_dead(self, *, batch_size: int = 20, memory_kind: str | None = None) -> int:
        ...


class PostgresVectorOutboxStoreImpl:
    """Postgres outbox 实现。"""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with psycopg.connect(self._database_url) as conn:
            conn.execute(_OUTBOX_SCHEMA_SQL)
            conn.commit()
        self._schema_ready = True

    def pending_count(self, *, memory_kind: str | None = None) -> int:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            if memory_kind is None:
                row = conn.execute(
                    """
                    SELECT count(*) AS total
                    FROM memory_vector_outbox
                    WHERE status IN ('pending', 'processing')
                      AND next_retry_at <= now()
                    """,
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT count(*) AS total
                    FROM memory_vector_outbox
                    WHERE status IN ('pending', 'processing')
                      AND next_retry_at <= now()
                      AND memory_kind = %s
                    """,
                    (memory_kind,),
                ).fetchone()
        return int(row["total"]) if row else 0

    def status_counts(self) -> dict[str, dict[str, int]]:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT memory_kind, status, count(*) AS total
                FROM memory_vector_outbox
                WHERE status IN ('pending', 'processing', 'dead')
                GROUP BY memory_kind, status
                """
            ).fetchall()

        counts: dict[str, dict[str, int]] = {}
        for row in rows:
            kind = str(row["memory_kind"])
            status = str(row["status"])
            counts.setdefault(kind, {"pending": 0, "processing": 0, "dead": 0})
            counts[kind][status] = int(row["total"])
        return counts

    def enqueue_upsert(
        self,
        *,
        memory_kind: VectorOutboxKind,
        memory_id: str,
        user_id: str,
        session_id: str | None,
        collection_name: str,
        vector: list[float],
        max_attempts: int,
    ) -> None:
        self._upsert_entry(
            memory_kind=memory_kind,
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
            collection_name=collection_name,
            op="upsert",
            vector=vector,
            max_attempts=max_attempts,
        )

    def enqueue_delete(
        self,
        *,
        memory_kind: VectorOutboxKind,
        memory_id: str,
        user_id: str,
        session_id: str | None,
        collection_name: str,
        max_attempts: int,
    ) -> None:
        self._upsert_entry(
            memory_kind=memory_kind,
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
            collection_name=collection_name,
            op="delete",
            vector=None,
            max_attempts=max_attempts,
        )

    def _upsert_entry(
        self,
        *,
        memory_kind: str,
        memory_id: str,
        user_id: str,
        session_id: str | None,
        collection_name: str,
        op: str,
        vector: list[float] | None,
        max_attempts: int,
    ) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                """
                INSERT INTO memory_vector_outbox (
                    memory_kind, memory_id, user_id, session_id,
                    collection_name, op, vector, status, attempts,
                    max_attempts, next_retry_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', 0, %s, now(), now())
                ON CONFLICT (memory_kind, memory_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    session_id = EXCLUDED.session_id,
                    collection_name = EXCLUDED.collection_name,
                    op = EXCLUDED.op,
                    vector = EXCLUDED.vector,
                    status = 'pending',
                    attempts = 0,
                    max_attempts = EXCLUDED.max_attempts,
                    last_error = NULL,
                    next_retry_at = now(),
                    updated_at = now()
                """,
                (
                    memory_kind,
                    memory_id,
                    user_id,
                    session_id,
                    collection_name,
                    op,
                    Jsonb(vector) if vector is not None else None,
                    max_attempts,
                ),
            )
            conn.commit()

    def claim_pending(self, *, batch_size: int = 20) -> list[VectorOutboxRecord]:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                UPDATE memory_vector_outbox AS o
                SET status = 'processing', updated_at = now()
                WHERE o.id IN (
                    SELECT id
                    FROM memory_vector_outbox
                    WHERE status = 'pending'
                      AND next_retry_at <= now()
                    ORDER BY next_retry_at ASC, id ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                )
                RETURNING id, memory_kind, memory_id, user_id, session_id,
                          collection_name, op, vector, status, attempts, max_attempts
                """,
                (batch_size,),
            ).fetchall()
            conn.commit()
        return [_row_to_record(row) for row in rows]

    def mark_done(self, entry_id: int) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                """
                UPDATE memory_vector_outbox
                SET status = 'done', updated_at = now(), last_error = NULL
                WHERE id = %s
                """,
                (entry_id,),
            )
            conn.commit()

    def mark_failed(self, entry_id: int, error: str, *, max_attempts: int) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                UPDATE memory_vector_outbox
                SET attempts = attempts + 1,
                    last_error = %s,
                    updated_at = now(),
                    status = CASE
                        WHEN attempts + 1 >= %s THEN 'dead'
                        ELSE 'pending'
                    END,
                    next_retry_at = CASE
                        WHEN attempts + 1 >= %s THEN next_retry_at
                        ELSE now() + ((attempts + 1) * interval '30 seconds')
                    END
                WHERE id = %s
                RETURNING attempts, status
                """,
                (error[:2000], max_attempts, max_attempts, entry_id),
            ).fetchone()
            conn.commit()
        _ = row

    def reclaim_stale_processing(self, *, timeout_seconds: int) -> int:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                UPDATE memory_vector_outbox
                SET status = 'pending',
                    updated_at = now(),
                    last_error = coalesce(last_error, '') || ' [reclaimed stale processing]'
                WHERE status = 'processing'
                  AND updated_at < now() - (%s * interval '1 second')
                """,
                (timeout_seconds,),
            )
            conn.commit()
            return row.rowcount or 0

    def replay_dead(
        self,
        *,
        batch_size: int = 20,
        memory_kind: str | None = None,
    ) -> int:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            if memory_kind is None:
                row = conn.execute(
                    """
                    UPDATE memory_vector_outbox AS o
                    SET status = 'pending',
                        attempts = 0,
                        next_retry_at = now(),
                        updated_at = now(),
                        last_error = NULL
                    WHERE o.id IN (
                        SELECT id FROM memory_vector_outbox
                        WHERE status = 'dead'
                        ORDER BY updated_at ASC
                        LIMIT %s
                    )
                    """,
                    (batch_size,),
                )
            else:
                row = conn.execute(
                    """
                    UPDATE memory_vector_outbox AS o
                    SET status = 'pending',
                        attempts = 0,
                        next_retry_at = now(),
                        updated_at = now(),
                        last_error = NULL
                    WHERE o.id IN (
                        SELECT id FROM memory_vector_outbox
                        WHERE status = 'dead' AND memory_kind = %s
                        ORDER BY updated_at ASC
                        LIMIT %s
                    )
                    """,
                    (memory_kind, batch_size),
                )
            conn.commit()
            return row.rowcount or 0


def create_postgres_outbox_store(config: MemoryConfig) -> PostgresVectorOutboxStoreImpl:
    if not config.database_url:
        raise ValueError("未配置 DATABASE_URL，无法启用 Postgres vector outbox")
    return PostgresVectorOutboxStoreImpl(config.database_url)


def _row_to_record(row: dict[str, Any]) -> VectorOutboxRecord:
    raw_vector = row.get("vector")
    vector: list[float] | None = None
    if raw_vector is not None:
        if isinstance(raw_vector, str):
            vector = json.loads(raw_vector)
        else:
            vector = list(raw_vector)
    return VectorOutboxRecord(
        id=int(row["id"]),
        memory_kind=str(row["memory_kind"]),
        memory_id=str(row["memory_id"]),
        user_id=str(row["user_id"]),
        session_id=row.get("session_id"),
        collection_name=str(row["collection_name"]),
        op=str(row["op"]),
        vector=vector,
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
    )
