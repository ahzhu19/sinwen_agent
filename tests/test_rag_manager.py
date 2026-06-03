"""RAG manager tests."""

from rag.chunker import MarkdownChunker
from rag.manager import RagManager
from rag.storage import InMemoryRagStore
from tests.rag_fakes import FakeConverter, FakeEmbeddingProvider, FakeLLM, FakeVectorStore


def test_rag_manager_ingest_search_and_answer() -> None:
    store = InMemoryRagStore()
    manager = RagManager(
        store=store,
        converter=FakeConverter("# Milvus\n\nMilvus stores vector chunks."),
        chunker=MarkdownChunker(target_tokens=20, max_tokens=40, overlap_tokens=5),
        vector_store=FakeVectorStore(),
        embedding_provider=FakeEmbeddingProvider(),
        llm=FakeLLM("Milvus stores vector chunks. [Source 1]"),
    )

    document = manager.ingest("/tmp/milvus.md")
    results = manager.search("vector chunks", top_k=1)
    answer = manager.answer("Milvus 存什么？", top_k=1)

    assert document.id == results[0].document.id
    assert "Milvus stores vector chunks" in answer.answer


def test_rag_manager_delete_reindex_and_stats() -> None:
    store = InMemoryRagStore()
    vector_store = FakeVectorStore()
    manager = RagManager(
        store=store,
        converter=FakeConverter("# Doc\n\nChunk content for stats."),
        chunker=MarkdownChunker(target_tokens=20, max_tokens=40, overlap_tokens=5),
        vector_store=vector_store,
        embedding_provider=FakeEmbeddingProvider(),
        llm=FakeLLM(),
    )

    document = manager.ingest("/tmp/doc.md")
    assert manager.stats()["document_count"] == 1
    assert len(vector_store.records) >= 1

    manager.reindex(document.id)
    assert manager.stats()["indexed_chunk_count"] >= 1

    manager.delete(document.id)
    assert manager.stats()["document_count"] == 0
    assert not any(
        record["document_id"] == document.id for record in vector_store.records.values()
    )
