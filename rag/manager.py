"""High-level RAG facade."""

from __future__ import annotations

from typing import Any

import os
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from core.llm import BaseLLM
from memory.config import MemoryConfig
from memory.embedding import create_embedding_provider

from .chunker import MarkdownChunker
from .config import RagConfig
from .converter import DocumentConverter, MarkItDownConverter
from .generator import RagGenerator
from .ingestion import RagIngestionService
from .models import INGESTION_INDEXED, BatchIngestResult, RagAnswer, RagDocument, RagSearchResult
from .retriever import RagRetriever
from .outbox_store import create_rag_outbox_store
from .storage import RagStore, create_rag_store
from .vector_store import MilvusRagVectorStore, RagVectorStore


class RagManager:
    def __init__(
        self,
        config: RagConfig | None = None,
        store: RagStore | None = None,
        converter: DocumentConverter | None = None,
        chunker: MarkdownChunker | None = None,
        vector_store: RagVectorStore | None = None,
        embedding_provider: Any | None = None,
        llm: Any | None = None,
    ) -> None:
        self.config = config or RagConfig.from_env()
        self.store = store or create_rag_store(self.config.database_url)
        self.converter = converter or MarkItDownConverter()
        self.chunker = chunker or MarkdownChunker(
            target_tokens=self.config.target_chunk_tokens,
            max_tokens=self.config.max_chunk_tokens,
            overlap_tokens=self.config.overlap_tokens,
        )
        self.vector_store = vector_store or MilvusRagVectorStore(
            uri=self.config.milvus_uri,
            collection_name=self.config.rag_milvus_collection(),
            metric_type=self.config.metric_type,
            timeout=self.config.timeout,
        )
        self.embedding_provider = embedding_provider or create_embedding_provider(
            MemoryConfig.from_env()
        )
        self.llm = llm or BaseLLM()
        self._vector_outbox = (
            create_rag_outbox_store(self.config)
            if self.config.enable_rag_vector_outbox and self.config.database_url
            else None
        )

    def _ingestion_service(self) -> RagIngestionService:
        return RagIngestionService(
            converter=self.converter,
            chunker=self.chunker,
            store=self.store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            vector_outbox=self._vector_outbox,
            collection_name=self.config.rag_milvus_collection(),
            enable_vector_outbox=self.config.enable_rag_vector_outbox,
            vector_outbox_max_attempts=self.config.rag_vector_outbox_max_attempts,
        )

    def _retriever(self) -> RagRetriever:
        return RagRetriever(
            store=self.store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            llm=self.llm,
        )

    def ingest(
        self,
        source: str,
        source_type: str = "file",
        metadata: dict[str, Any] | None = None,
        *,
        source_uri: str | None = None,
    ) -> RagDocument:
        service = self._ingestion_service()
        return service.ingest(
            source=source,
            source_type=source_type,
            metadata=metadata,
            source_uri=source_uri,
        )

    def ingest_url(
        self,
        url: str,
        metadata: dict[str, Any] | None = None,
    ) -> RagDocument:
        """Download content from URL and ingest it."""
        with urllib.request.urlopen(url, timeout=self.config.timeout) as response:
            raw = response.read()
        suffix = _url_suffix(url)
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            return self.ingest(
                source=tmp_path,
                source_type="url",
                metadata=metadata,
                source_uri=url,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def ingest_directory(
        self,
        path: str,
        pattern: str = "**/*.md",
        metadata: dict[str, Any] | None = None,
    ) -> BatchIngestResult:
        """Ingest all files matching a glob pattern from a directory."""
        directory = Path(path)
        if not directory.is_dir():
            raise NotADirectoryError(f"不是有效目录: {path}")
        documents: list[RagDocument] = []
        errors: list[str] = []
        for file_path in sorted(directory.glob(pattern)):
            if not file_path.is_file():
                continue
            try:
                doc = self.ingest(
                    source=str(file_path),
                    source_type="file",
                    metadata=metadata,
                )
                documents.append(doc)
            except Exception as exc:
                errors.append(f"{file_path}: {exc}")
        return BatchIngestResult(documents=documents, errors=errors)

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "direct",
        rerank: str | bool | None = None,
    ) -> list[RagSearchResult]:
        return self._retriever().search(
            query=query, top_k=top_k, strategy=strategy, rerank=rerank
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "direct",
        rerank: str | bool | None = None,
    ) -> RagAnswer:
        sources = self.search(query=query, top_k=top_k, strategy=strategy, rerank=rerank)
        return RagGenerator(self.llm).answer(query=query, sources=sources)

    def list_documents(self, limit: int = 50) -> list[RagDocument]:
        return self.store.list_documents(limit=limit)

    def delete(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        self.store.delete_document(document_id)

    def reindex(self, document_id: str) -> RagDocument:
        document = self.store.get_document(document_id)
        chunks = self.store.get_chunks_for_document(document_id)
        if not chunks:
            raise ValueError(f"文档 {document_id} 没有可重建索引的 chunk")

        vectors = self.embedding_provider.embed_batch([chunk.content for chunk in chunks])
        self.vector_store.delete_document(document_id)
        self.vector_store.upsert_many(
            [(chunk, vector, document) for chunk, vector in zip(chunks, vectors, strict=True)]
        )
        self.store.mark_chunks_indexed([chunk.id for chunk in chunks])
        self.store.update_document_status(document_id, INGESTION_INDEXED)
        return self.store.get_document(document_id)

    def stats(self) -> dict[str, Any]:
        document_count = len(self.store.list_documents(limit=10_000))
        chunk_count, indexed_count = self.store.count_chunks()
        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "indexed_chunk_count": indexed_count,
            "collection": self.config.rag_milvus_collection(),
        }


def _url_suffix(url: str) -> str:
    """Extract file suffix from URL path, defaulting to .html."""
    path = urlparse(url).path
    suffix = Path(path).suffix
    return suffix if suffix else ".html"

