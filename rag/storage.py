"""RAG metadata storage."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import (
    INGESTION_PENDING,
    IngestionRun,
    RagChunk,
    RagDocument,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY,
    source_uri TEXT NOT NULL,
    source_type TEXT NOT NULL,
    title TEXT,
    mime_type TEXT,
    content_hash TEXT NOT NULL,
    markdown TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    heading_path JSONB NOT NULL DEFAULT '[]'::jsonb,
    token_count INTEGER NOT NULL,
    char_start INTEGER,
    char_end INTEGER,
    indexed BOOLEAN NOT NULL DEFAULT false,
    indexed_at TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rag_ingestion_runs (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES rag_documents(id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_rag_documents_content_hash ON rag_documents (content_hash);
CREATE INDEX IF NOT EXISTS idx_rag_documents_status ON rag_documents (status);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_index ON rag_chunks (document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_indexed ON rag_chunks (indexed);
CREATE INDEX IF NOT EXISTS idx_rag_ingestion_runs_document_started
    ON rag_ingestion_runs (document_id, started_at);
"""


class RagStore(Protocol):
    def start_ingestion(self, metadata: dict[str, Any]) -> IngestionRun:
        ...

    def finish_ingestion(self, run_id: str, status: str, error_message: str | None = None) -> None:
        ...

    def create_document(
        self,
        source_uri: str,
        source_type: str,
        title: str | None,
        mime_type: str | None,
        content_hash: str,
        markdown: str,
        status: str,
        metadata: dict[str, Any],
        run_id: str | None,
    ) -> RagDocument:
        ...

    def update_document_status(self, document_id: str, status: str) -> None:
        ...

    def get_document(self, document_id: str) -> RagDocument:
        ...

    def replace_chunks(self, document_id: str, chunks: list[RagChunk]) -> None:
        ...

    def get_chunks(self, chunk_ids: list[str]) -> list[RagChunk]:
        ...

    def get_chunks_for_document(self, document_id: str) -> list[RagChunk]:
        ...

    def mark_chunks_indexed(self, chunk_ids: list[str]) -> None:
        ...

    def delete_document(self, document_id: str) -> None:
        ...

    def list_documents(self, limit: int = 50) -> list[RagDocument]:
        ...


class InMemoryRagStore:
    def __init__(self) -> None:
        self.documents: dict[str, RagDocument] = {}
        self.chunks: dict[str, RagChunk] = {}
        self.runs: dict[str, IngestionRun] = {}

    def start_ingestion(self, metadata: dict[str, Any]) -> IngestionRun:
        run = IngestionRun(
            id=str(uuid4()),
            document_id=None,
            status=INGESTION_PENDING,
            metadata=dict(metadata),
            started_at=datetime.now(timezone.utc),
        )
        self.runs[run.id] = run
        return run

    def finish_ingestion(self, run_id: str, status: str, error_message: str | None = None) -> None:
        run = self.runs[run_id]
        self.runs[run_id] = replace(
            run,
            status=status,
            error_message=error_message,
            finished_at=datetime.now(timezone.utc),
        )

    def create_document(
        self,
        source_uri: str,
        source_type: str,
        title: str | None,
        mime_type: str | None,
        content_hash: str,
        markdown: str,
        status: str,
        metadata: dict[str, Any],
        run_id: str | None,
    ) -> RagDocument:
        now = datetime.now(timezone.utc)
        document = RagDocument(
            id=str(uuid4()),
            source_uri=source_uri,
            source_type=source_type,
            title=title,
            mime_type=mime_type,
            content_hash=content_hash,
            markdown=markdown,
            status=status,
            metadata=dict(metadata),
            created_at=now,
            updated_at=now,
        )
        self.documents[document.id] = document
        if run_id is not None:
            run = self.runs[run_id]
            self.runs[run_id] = replace(run, document_id=document.id)
        return document

    def update_document_status(self, document_id: str, status: str) -> None:
        document = self.documents[document_id]
        self.documents[document_id] = replace(
            document,
            status=status,
            updated_at=datetime.now(timezone.utc),
        )

    def get_document(self, document_id: str) -> RagDocument:
        return self.documents[document_id]

    def replace_chunks(self, document_id: str, chunks: list[RagChunk]) -> None:
        for chunk_id, chunk in list(self.chunks.items()):
            if chunk.document_id == document_id:
                self.chunks.pop(chunk_id)
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    def get_chunks(self, chunk_ids: list[str]) -> list[RagChunk]:
        return [self.chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self.chunks]

    def get_chunks_for_document(self, document_id: str) -> list[RagChunk]:
        chunks = [chunk for chunk in self.chunks.values() if chunk.document_id == document_id]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def mark_chunks_indexed(self, chunk_ids: list[str]) -> None:
        now = datetime.now(timezone.utc)
        for chunk_id in chunk_ids:
            if chunk_id in self.chunks:
                self.chunks[chunk_id] = replace(
                    self.chunks[chunk_id], indexed=True, indexed_at=now
                )

    def delete_document(self, document_id: str) -> None:
        self.documents.pop(document_id, None)
        for chunk_id, chunk in list(self.chunks.items()):
            if chunk.document_id == document_id:
                self.chunks.pop(chunk_id)

    def list_documents(self, limit: int = 50) -> list[RagDocument]:
        documents = sorted(
            self.documents.values(),
            key=lambda document: document.updated_at or document.created_at,
            reverse=True,
        )
        return documents[:limit]


class PostgresRagStore:
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

    def start_ingestion(self, metadata: dict[str, Any]) -> IngestionRun:
        self.ensure_schema()
        run_id = str(uuid4())
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                INSERT INTO rag_ingestion_runs (id, status, metadata)
                VALUES (%s, %s, %s)
                RETURNING id, document_id, status, error_message, metadata, started_at, finished_at
                """,
                (run_id, INGESTION_PENDING, Jsonb(dict(metadata))),
            ).fetchone()
            conn.commit()
        assert row is not None
        return _row_to_run(row)

    def finish_ingestion(self, run_id: str, status: str, error_message: str | None = None) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                """
                UPDATE rag_ingestion_runs
                SET status = %s, error_message = %s, finished_at = now()
                WHERE id = %s
                """,
                (status, error_message, run_id),
            )
            conn.commit()

    def create_document(
        self,
        source_uri: str,
        source_type: str,
        title: str | None,
        mime_type: str | None,
        content_hash: str,
        markdown: str,
        status: str,
        metadata: dict[str, Any],
        run_id: str | None,
    ) -> RagDocument:
        self.ensure_schema()
        document_id = str(uuid4())
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                INSERT INTO rag_documents (
                    id, source_uri, source_type, title, mime_type,
                    content_hash, markdown, status, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, source_uri, source_type, title, mime_type, content_hash,
                          markdown, status, metadata, created_at, updated_at
                """,
                (
                    document_id,
                    source_uri,
                    source_type,
                    title,
                    mime_type,
                    content_hash,
                    markdown,
                    status,
                    Jsonb(dict(metadata)),
                ),
            ).fetchone()
            if run_id is not None:
                conn.execute(
                    "UPDATE rag_ingestion_runs SET document_id = %s WHERE id = %s",
                    (document_id, run_id),
                )
            conn.commit()
        assert row is not None
        return _row_to_document(row)

    def update_document_status(self, document_id: str, status: str) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                """
                UPDATE rag_documents
                SET status = %s, updated_at = now()
                WHERE id = %s
                """,
                (status, document_id),
            )
            conn.commit()

    def get_document(self, document_id: str) -> RagDocument:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            row = conn.execute(
                """
                SELECT id, source_uri, source_type, title, mime_type, content_hash,
                       markdown, status, metadata, created_at, updated_at
                FROM rag_documents
                WHERE id = %s
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            raise KeyError(document_id)
        return _row_to_document(row)

    def replace_chunks(self, document_id: str, chunks: list[RagChunk]) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            conn.execute("DELETE FROM rag_chunks WHERE document_id = %s", (document_id,))
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO rag_chunks (
                        id, document_id, chunk_index, content, heading_path,
                        token_count, char_start, char_end, indexed, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk.id,
                        document_id,
                        chunk.chunk_index,
                        chunk.content,
                        Jsonb(list(chunk.heading_path)),
                        chunk.token_count,
                        chunk.char_start,
                        chunk.char_end,
                        chunk.indexed,
                        Jsonb(dict(chunk.metadata)),
                    ),
                )
            conn.commit()

    def get_chunks(self, chunk_ids: list[str]) -> list[RagChunk]:
        if not chunk_ids:
            return []
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, chunk_index, content, heading_path,
                       token_count, char_start, char_end, indexed, indexed_at, metadata
                FROM rag_chunks
                WHERE id = ANY(%s)
                """,
                (chunk_ids,),
            ).fetchall()
        order = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
        chunks = [_row_to_chunk(row) for row in rows]
        chunks.sort(key=lambda chunk: order.get(chunk.id, len(chunk_ids)))
        return chunks

    def get_chunks_for_document(self, document_id: str) -> list[RagChunk]:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, chunk_index, content, heading_path,
                       token_count, char_start, char_end, indexed, indexed_at, metadata
                FROM rag_chunks
                WHERE document_id = %s
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def mark_chunks_indexed(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            conn.execute(
                """
                UPDATE rag_chunks
                SET indexed = true, indexed_at = now()
                WHERE id = ANY(%s)
                """,
                (chunk_ids,),
            )
            conn.commit()

    def delete_document(self, document_id: str) -> None:
        self.ensure_schema()
        with psycopg.connect(self._database_url) as conn:
            conn.execute("DELETE FROM rag_documents WHERE id = %s", (document_id,))
            conn.commit()

    def list_documents(self, limit: int = 50) -> list[RagDocument]:
        self.ensure_schema()
        with psycopg.connect(self._database_url, row_factory=dict_row) as conn:
            rows = conn.execute(
                """
                SELECT id, source_uri, source_type, title, mime_type, content_hash,
                       markdown, status, metadata, created_at, updated_at
                FROM rag_documents
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [_row_to_document(row) for row in rows]


def create_rag_store(database_url: str | None) -> PostgresRagStore:
    if not database_url:
        raise ValueError("未配置 DATABASE_URL，无法启用 RAG PostgreSQL 存储")
    return PostgresRagStore(database_url)


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return json.loads(value)
    return dict(value or {})


def _parse_heading_path(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        parsed = json.loads(value)
        return [str(item) for item in parsed]
    return []


def _row_to_document(row: dict[str, Any]) -> RagDocument:
    return RagDocument(
        id=str(row["id"]),
        source_uri=row["source_uri"],
        source_type=row["source_type"],
        title=row["title"],
        mime_type=row["mime_type"],
        content_hash=row["content_hash"],
        markdown=row["markdown"],
        status=row["status"],
        metadata=_parse_json(row["metadata"]),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_chunk(row: dict[str, Any]) -> RagChunk:
    return RagChunk(
        id=str(row["id"]),
        document_id=str(row["document_id"]),
        chunk_index=int(row["chunk_index"]),
        content=row["content"],
        heading_path=_parse_heading_path(row["heading_path"]),
        token_count=int(row["token_count"]),
        char_start=row.get("char_start"),
        char_end=row.get("char_end"),
        indexed=bool(row.get("indexed", False)),
        indexed_at=row.get("indexed_at"),
        metadata=_parse_json(row.get("metadata")),
    )


def _row_to_run(row: dict[str, Any]) -> IngestionRun:
    return IngestionRun(
        id=str(row["id"]),
        document_id=str(row["document_id"]) if row.get("document_id") else None,
        status=row["status"],
        error_message=row.get("error_message"),
        metadata=_parse_json(row.get("metadata")),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )
