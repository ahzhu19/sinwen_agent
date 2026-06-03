"""RAG storage tests."""

from rag.models import INGESTION_CHUNKED, INGESTION_INDEXED, RagChunk
from rag.storage import InMemoryRagStore


def test_store_creates_document_run_and_chunks() -> None:
    store = InMemoryRagStore()
    run = store.start_ingestion(metadata={"source": "test"})

    document = store.create_document(
        source_uri="/tmp/guide.md",
        source_type="file",
        title="guide.md",
        mime_type="text/markdown",
        content_hash="hash1",
        markdown="# Guide",
        status=INGESTION_CHUNKED,
        metadata={"team": "agent"},
        run_id=run.id,
    )
    chunks = [
        RagChunk(
            id="chunk1",
            document_id=document.id,
            chunk_index=0,
            content="# Guide\nBody",
            heading_path=["Guide"],
            token_count=3,
        )
    ]

    store.replace_chunks(document.id, chunks)

    assert store.get_document(document.id) == document
    assert store.get_chunks(["chunk1"])[0].content == "# Guide\nBody"
    assert store.get_chunks_for_document(document.id)[0].heading_path == ["Guide"]


def test_store_marks_chunks_indexed() -> None:
    store = InMemoryRagStore()
    document = store.create_document(
        source_uri="/tmp/guide.md",
        source_type="file",
        title="guide.md",
        mime_type="text/markdown",
        content_hash="hash1",
        markdown="# Guide",
        status=INGESTION_CHUNKED,
        metadata={},
        run_id=None,
    )
    store.replace_chunks(
        document.id,
        [
            RagChunk(
                id="chunk1",
                document_id=document.id,
                chunk_index=0,
                content="Body",
                heading_path=[],
                token_count=1,
            )
        ],
    )

    store.mark_chunks_indexed(["chunk1"])
    store.update_document_status(document.id, INGESTION_INDEXED)

    chunk = store.get_chunks(["chunk1"])[0]
    assert chunk.indexed is True
    assert chunk.indexed_at is not None
    assert store.get_document(document.id).status == INGESTION_INDEXED
