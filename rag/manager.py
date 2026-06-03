"""High-level RAG facade."""

from __future__ import annotations

from typing import Any

from core.llm import BaseLLM
from memory.config import MemoryConfig
from memory.embedding import create_embedding_provider

from .chunker import MarkdownChunker
from .config import RagConfig
from .converter import DocumentConverter, MarkItDownConverter
from .generator import RagGenerator
from .ingestion import RagIngestionService
from .models import INGESTION_INDEXED, RagAnswer, RagDocument, RagSearchResult
from .retriever import RagRetriever
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
            collection_name=self.config.collection_name,
            metric_type=self.config.metric_type,
            timeout=self.config.timeout,
        )
        self.embedding_provider = embedding_provider or create_embedding_provider(
            MemoryConfig.from_env()
        )
        self.llm = llm or BaseLLM()

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
    ) -> RagDocument:
        service = RagIngestionService(
            converter=self.converter,
            chunker=self.chunker,
            store=self.store,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
        )
        return service.ingest(source=source, source_type=source_type, metadata=metadata)

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "direct",
    ) -> list[RagSearchResult]:
        return self._retriever().search(query=query, top_k=top_k, strategy=strategy)

    def answer(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "direct",
    ) -> RagAnswer:
        sources = self.search(query=query, top_k=top_k, strategy=strategy)
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
        documents = self.store.list_documents(limit=10_000)
        chunk_count = 0
        indexed_count = 0
        for document in documents:
            chunks = self.store.get_chunks_for_document(document.id)
            chunk_count += len(chunks)
            indexed_count += sum(1 for chunk in chunks if chunk.indexed)
        return {
            "document_count": len(documents),
            "chunk_count": chunk_count,
            "indexed_chunk_count": indexed_count,
            "collection": self.config.collection_name,
        }
