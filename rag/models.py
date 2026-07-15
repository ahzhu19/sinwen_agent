"""RAG domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

INGESTION_PENDING = "pending"
INGESTION_CONVERTED = "converted"
INGESTION_CHUNKED = "chunked"
INGESTION_EMBEDDED = "embedded"
INGESTION_INDEXED = "indexed"
INGESTION_FAILED = "failed"


@dataclass(frozen=True)
class RagDocument:
    id: str
    source_uri: str
    source_type: str
    title: str | None
    mime_type: str | None
    content_hash: str
    markdown: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class RagChunk:
    id: str
    document_id: str
    chunk_index: int
    content: str
    heading_path: list[str]
    token_count: int
    char_start: int | None = None
    char_end: int | None = None
    indexed: bool = False
    indexed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionRun:
    id: str
    document_id: str | None
    status: str
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True)
class RagSearchResult:
    chunk: RagChunk
    document: RagDocument
    score: float


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[RagSearchResult]

@dataclass(frozen=True)
class BatchIngestResult:
    """Result of batch ingestion (directory)."""

    documents: list[RagDocument]
    errors: list[str]

    @property
    def success_count(self) -> int:
        return len(self.documents)

    @property
    def error_count(self) -> int:
        return len(self.errors)

