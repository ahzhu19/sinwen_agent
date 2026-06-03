"""RAG retrieval and generation tests."""

from rag.chunker import MarkdownChunker
from rag.generator import RagGenerator
from rag.ingestion import RagIngestionService
from rag.retriever import RagRetriever
from rag.storage import InMemoryRagStore
from tests.rag_fakes import FakeConverter, FakeEmbeddingProvider, FakeLLM, FakeVectorStore


def _ingest_fixture():
    store = InMemoryRagStore()
    vector_store = FakeVectorStore()
    embeddings = FakeEmbeddingProvider()
    ingestion = RagIngestionService(
        converter=FakeConverter("# Milvus\n\nMilvus uses collections for vectors."),
        chunker=MarkdownChunker(target_tokens=20, max_tokens=40, overlap_tokens=5),
        store=store,
        vector_store=vector_store,
        embedding_provider=embeddings,
    )
    document = ingestion.ingest("/tmp/milvus.md")
    return store, vector_store, embeddings, document


def test_retriever_hydrates_vector_hits_with_chunks_and_documents() -> None:
    store, vector_store, embeddings, document = _ingest_fixture()
    retriever = RagRetriever(store=store, vector_store=vector_store, embedding_provider=embeddings)

    results = retriever.search("Milvus collections", top_k=1)

    assert len(results) == 1
    assert results[0].document.id == document.id
    assert "Milvus uses collections" in results[0].chunk.content


def test_generator_builds_answer_with_sources() -> None:
    store, vector_store, embeddings, _ = _ingest_fixture()
    retriever = RagRetriever(store=store, vector_store=vector_store, embedding_provider=embeddings)
    generator = RagGenerator(llm=FakeLLM("Milvus 使用 collections。[Source 1]"))

    results = retriever.search("Milvus 怎么存向量？", top_k=1)
    answer = generator.answer("Milvus 怎么存向量？", results)

    assert "Milvus 使用 collections" in answer.answer
    assert answer.sources == results
