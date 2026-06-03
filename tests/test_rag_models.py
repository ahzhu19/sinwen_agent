"""RAG model and config tests."""

from rag.config import RagConfig
from rag.models import (
    INGESTION_INDEXED,
    RagAnswer,
    RagChunk,
    RagDocument,
    RagSearchResult,
)


def test_rag_config_has_document_rag_defaults() -> None:
    config = RagConfig()

    assert config.collection_name == "hello_agents_rag_chunks"
    assert config.target_chunk_tokens == 500
    assert config.max_chunk_tokens == 800
    assert config.overlap_tokens == 80


def test_rag_models_store_source_and_chunk_metadata() -> None:
    document = RagDocument(
        id="doc1",
        source_uri="/tmp/guide.md",
        source_type="file",
        title="Guide",
        mime_type="text/markdown",
        content_hash="hash1",
        markdown="# Guide\nContent",
        status=INGESTION_INDEXED,
        metadata={"owner": "docs"},
    )
    chunk = RagChunk(
        id="chunk1",
        document_id="doc1",
        chunk_index=0,
        content="# Guide\nContent",
        heading_path=["Guide"],
        token_count=3,
        char_start=0,
        char_end=15,
        indexed=True,
        metadata={"section": "intro"},
    )
    result = RagSearchResult(chunk=chunk, document=document, score=0.92)
    answer = RagAnswer(answer="Use Docker. [Source 1]", sources=[result])

    assert answer.sources[0].chunk.id == "chunk1"
    assert answer.sources[0].document.source_uri == "/tmp/guide.md"
    assert answer.answer.endswith("[Source 1]")
