"""RAG document ingestion orchestration."""

from __future__ import annotations

import hashlib
from typing import Any

from .chunker import MarkdownChunker
from .converter import DocumentConverter
from .models import (
    INGESTION_CHUNKED,
    INGESTION_FAILED,
    INGESTION_INDEXED,
    RagDocument,
)
from .storage import RagStore
from .vector_store import RagVectorStore


class RagIngestionService:
    def __init__(
        self,
        converter: DocumentConverter,
        chunker: MarkdownChunker,
        store: RagStore,
        vector_store: RagVectorStore,
        embedding_provider: Any,
    ) -> None:
        self._converter = converter
        self._chunker = chunker
        self._store = store
        self._vector_store = vector_store
        self._embeddings = embedding_provider

    def ingest(
        self,
        source: str,
        source_type: str = "file",
        metadata: dict[str, Any] | None = None,
    ) -> RagDocument:
        metadata = dict(metadata or {})
        run = self._store.start_ingestion({"source": source, "source_type": source_type})
        try:
            converted = self._converter.convert(source)
            content_hash = hashlib.sha256(converted.markdown.encode()).hexdigest()
            document = self._store.create_document(
                source_uri=source,
                source_type=source_type,
                title=converted.title,
                mime_type=converted.mime_type,
                content_hash=content_hash,
                markdown=converted.markdown,
                status=INGESTION_CHUNKED,
                metadata=metadata,
                run_id=run.id,
            )
            chunks = self._chunker.chunk(converted.markdown, document_id=document.id)
            self._store.replace_chunks(document.id, chunks)

            vectors = self._embeddings.embed_batch([chunk.content for chunk in chunks])
            self._vector_store.upsert_many(
                [(chunk, vector, document) for chunk, vector in zip(chunks, vectors, strict=True)]
            )
            self._store.mark_chunks_indexed([chunk.id for chunk in chunks])
            self._store.update_document_status(document.id, INGESTION_INDEXED)
            self._store.finish_ingestion(run.id, INGESTION_INDEXED)
            return self._store.get_document(document.id)
        except Exception as exc:
            self._store.finish_ingestion(run.id, INGESTION_FAILED, str(exc))
            raise
