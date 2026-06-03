"""RAG ingestion workflow tests."""

from rag.chunker import MarkdownChunker
from rag.ingestion import RagIngestionService
from rag.models import INGESTION_INDEXED
from rag.storage import InMemoryRagStore
from tests.rag_fakes import FakeConverter, FakeEmbeddingProvider, FakeVectorStore


def test_ingestion_converts_chunks_embeds_and_indexes_document() -> None:
    store = InMemoryRagStore()
    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddingProvider()
    service = RagIngestionService(
        converter=FakeConverter("# Guide\n\nMilvus setup instructions"),
        chunker=MarkdownChunker(target_tokens=20, max_tokens=40, overlap_tokens=5),
        store=store,
        vector_store=vector_store,
        embedding_provider=embeddings,
    )

    document = service.ingest("/tmp/guide.md", metadata={"team": "agent"})

    assert document.status == INGESTION_INDEXED
    assert document.markdown.startswith("# Guide")
    chunks = store.get_chunks_for_document(document.id)
    assert len(chunks) == 1
    assert chunks[0].indexed is True
    assert chunks[0].id in vector_store.records
    assert embeddings.calls[0] == [chunks[0].content]


def test_ingestion_records_failed_run_when_converter_raises() -> None:
    class BrokenConverter:
        def convert(self, source: str):
            _ = source
            raise RuntimeError("cannot convert")

    store = InMemoryRagStore()
    service = RagIngestionService(
        converter=BrokenConverter(),
        chunker=MarkdownChunker(),
        store=store,
        vector_store=FakeVectorStore(),
        embedding_provider=FakeEmbeddingProvider(),
    )

    try:
        service.ingest("/tmp/broken.pdf")
    except RuntimeError:
        pass

    run = next(iter(store.runs.values()))
    assert run.status == "failed"
    assert run.error_message == "cannot convert"
