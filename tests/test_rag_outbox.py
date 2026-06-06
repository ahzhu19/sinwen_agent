"""RAG vector outbox tests."""

from __future__ import annotations

from rag.chunker import MarkdownChunker
from rag.ingestion import RagIngestionService
from rag.models import INGESTION_INDEXED
from rag.outbox_store import PostgresRagVectorOutboxStore
from rag.storage import InMemoryRagStore
from tests.rag_fakes import FakeConverter, FakeEmbeddingProvider, FakeVectorStore


class InMemoryRagOutbox(PostgresRagVectorOutboxStore):
    """进程内 RAG outbox fake（不连 Postgres）。"""

    def __init__(self) -> None:
        self._entries: dict[int, dict] = {}
        self._seq = 0
        self._schema_ready = True

    def ensure_schema(self) -> None:
        return None

    def enqueue_many(self, entries, *, max_attempts: int) -> None:
        for chunk_id, document_id, source_uri, collection_name, vector in entries:
            self._seq += 1
            self._entries[self._seq] = {
                "id": self._seq,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source_uri": source_uri,
                "collection_name": collection_name,
                "vector": vector,
                "status": "pending",
                "attempts": 0,
                "max_attempts": max_attempts,
            }

    def pending_count(self) -> int:
        return sum(1 for entry in self._entries.values() if entry["status"] == "pending")


def test_rag_ingestion_enqueues_vector_outbox_instead_of_direct_milvus() -> None:
    store = InMemoryRagStore()
    vector_store = FakeVectorStore()
    outbox = InMemoryRagOutbox()
    service = RagIngestionService(
        converter=FakeConverter("# Guide\n\nMilvus setup instructions"),
        chunker=MarkdownChunker(target_tokens=20, max_tokens=40, overlap_tokens=5),
        store=store,
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        vector_outbox=outbox,
        collection_name="rag_vectors",
        enable_vector_outbox=True,
    )

    document = service.ingest("/tmp/guide.md", metadata={"team": "agent"})

    assert document.status == INGESTION_INDEXED
    chunks = store.get_chunks_for_document(document.id)
    assert len(chunks) == 1
    assert chunks[0].indexed is False
    assert chunks[0].id not in vector_store.records
    assert outbox.pending_count() == 1
