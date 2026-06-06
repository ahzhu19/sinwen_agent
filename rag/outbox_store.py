"""RAG Milvus 向量 outbox（Postgres 持久化）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import RagConfig

RagOutboxStatus = Literal["pending", "processing", "done", "dead"]


@dataclass(frozen=True)
class RagVectorOutboxRecord:
    id: int
    chunk_id: str
    document_id: str
    source_uri: str
    collection_name: str
    vector: list[float]
    status: str
    attempts: int
    max_attempts: int


_OUTBOX_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rag_vector_outbox (
    id BIGSERIAL PRIMARY KEY,
    chunk_id UUID NOT NULL UNIQUE,
    document_id UUID NOT NULL,
    source_uri TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    vector JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 5,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_vector_outbox_pending
    ON rag_vector_outbox (status, updated_at)
    WHERE status IN ('pending', 'processing');
"""


class RagVectorOutboxStore(Protocol):
    def ensure_schema(self) -> None:
        ...

    def enqueue_many(
        self,
        entries: list[tuple[str, str, str, str, list[float]]],
        *,
        max_attempts: int,
    ) -> None:
        ...

    def claim_pending(self, *, batch_size: int = 20) -> list[RagVectorOutboxRecord]:
        ...

    def mark_done(self, entry_id: int) -> None:
        ...

    def mark_failed(self, entry_id: int, error: str, *, max_attempts: int) -> None:
        ...

    def reclaim_stale_processing(self, *, timeout_seconds: int) -> int:
        ...

    def replay_dead(self, *, batch_size: int = 20) -> int:
        ...

    def pending_count(self) -> int:
        ...

    def status_counts(self) -> dict[str, int]:
        ...


class PostgresRagVectorOutboxStore:
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

    def enqueue_many(
        self,
        entries: list[tuple[str, str, str, str, list[float]]],
        *,
        max_attempts: int,
    ) -> None:
        if not entries:
            return
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            for chunk_id, document_id, source_uri, collection_name, vector in entries:
                conn.execute(
                    """
                    INSERT INTO rag_vector_outbox (
                        chunk_id, document_id, source_uri, collection_name,
                        vector, status, attempts, max_attempts
                    )
                    VALUES (%s, %s, %s, %s, %s, 'pending', 0, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source_uri = EXCLUDED.source_uri,
                        collection_name = EXCLUDED.collection_name,
                        vector = EXCLUDED.vector,
                        status = 'pending',
                        attempts = 0,
                        last_error = NULL,
                        updated_at = now()
                    """,
                    (
                        chunk_id,
                        document_id,
                        source_uri,
                        collection_name,
                        Jsonb(vector),
                        max_attempts,
                    ),
                )
            conn.commit()

    def claim_pending(self, *, batch_size: int = 20) -> list[RagVectorOutboxRecord]:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                UPDATE rag_vector_outbox AS o
                SET status = 'processing', updated_at = now()
                WHERE o.id IN (
                    SELECT id FROM rag_vector_outbox
                    WHERE status = 'pending'
                    ORDER BY updated_at ASC, id ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                )
                RETURNING id, chunk_id, document_id, source_uri, collection_name,
                          vector, status, attempts, max_attempts
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
                UPDATE rag_vector_outbox
                SET status = 'done', updated_at = now(), last_error = NULL
                WHERE id = %s
                """,
                (entry_id,),
            )
            conn.commit()

    def mark_failed(self, entry_id: int, error: str, *, max_attempts: int) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            conn.execute(
                """
                UPDATE rag_vector_outbox
                SET attempts = attempts + 1,
                    last_error = %s,
                    updated_at = now(),
                    status = CASE
                        WHEN attempts + 1 >= %s THEN 'dead'
                        ELSE 'pending'
                    END
                WHERE id = %s
                """,
                (error[:2000], max_attempts, entry_id),
            )
            conn.commit()

    def reclaim_stale_processing(self, *, timeout_seconds: int) -> int:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            row = conn.execute(
                """
                UPDATE rag_vector_outbox
                SET status = 'pending', updated_at = now()
                WHERE status = 'processing'
                  AND updated_at < now() - (%s * interval '1 second')
                """,
                (timeout_seconds,),
            )
            conn.commit()
            return row.rowcount or 0

    def replay_dead(self, *, batch_size: int = 20) -> int:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            row = conn.execute(
                """
                UPDATE rag_vector_outbox AS o
                SET status = 'pending', attempts = 0, updated_at = now(), last_error = NULL
                WHERE o.id IN (
                    SELECT id FROM rag_vector_outbox
                    WHERE status = 'dead'
                    ORDER BY updated_at ASC
                    LIMIT %s
                )
                """,
                (batch_size,),
            )
            conn.commit()
            return row.rowcount or 0

    def pending_count(self) -> int:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                SELECT count(*) AS total FROM rag_vector_outbox
                WHERE status IN ('pending', 'processing')
                """,
            ).fetchone()
        return int(row["total"]) if row else 0

    def status_counts(self) -> dict[str, int]:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT status, count(*) AS total
                FROM rag_vector_outbox
                WHERE status IN ('pending', 'processing', 'dead')
                GROUP BY status
                """,
            ).fetchall()
        counts = {"pending": 0, "processing": 0, "dead": 0}
        for row in rows:
            counts[str(row["status"])] = int(row["total"])
        return counts


def create_rag_outbox_store(config: RagConfig) -> PostgresRagVectorOutboxStore:
    if not config.database_url:
        raise ValueError("未配置 DATABASE_URL，无法启用 RAG vector outbox")
    return PostgresRagVectorOutboxStore(config.database_url)


def _row_to_record(row: dict[str, Any]) -> RagVectorOutboxRecord:
    raw_vector = row.get("vector")
    if isinstance(raw_vector, str):
        vector = json.loads(raw_vector)
    else:
        vector = list(raw_vector)
    return RagVectorOutboxRecord(
        id=int(row["id"]),
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        source_uri=str(row["source_uri"]),
        collection_name=str(row["collection_name"]),
        vector=vector,
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        max_attempts=int(row["max_attempts"]),
    )
