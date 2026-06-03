"""RAG vector store tests."""

from rag.models import RagChunk, RagDocument
from tests.rag_fakes import FakeVectorStore


def test_fake_vector_store_upserts_and_searches_chunks() -> None:
    store = FakeVectorStore()
    document = RagDocument(
        id="doc1",
        source_uri="/tmp/guide.md",
        source_type="file",
        title="guide.md",
        mime_type="text/markdown",
        content_hash="hash",
        markdown="# Guide",
        status="indexed",
    )
    chunk = RagChunk(
        id="chunk1",
        document_id="doc1",
        chunk_index=0,
        content="Milvus setup",
        heading_path=[],
        token_count=2,
    )

    store.upsert_many([(chunk, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], document)])
    hits = store.search([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], limit=1)

    assert hits[0].chunk_id == "chunk1"
    assert hits[0].document_id == "doc1"
    assert hits[0].source_uri == "/tmp/guide.md"
