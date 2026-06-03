"""RagTool tests."""

from rag.models import RagAnswer, RagChunk, RagDocument, RagSearchResult
from tools.builtin.rag_tool import RagTool


class FakeRagManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.document = RagDocument(
            id="doc123456789",
            source_uri="/tmp/guide.md",
            source_type="file",
            title="guide.md",
            mime_type="text/markdown",
            content_hash="hash",
            markdown="# Guide",
            status="indexed",
        )
        self.chunk = RagChunk(
            id="chunk123456789",
            document_id=self.document.id,
            chunk_index=0,
            content="Milvus setup",
            heading_path=["Guide"],
            token_count=2,
            indexed=True,
        )
        self.result = RagSearchResult(chunk=self.chunk, document=self.document, score=0.9)

    def ingest(self, **kwargs):
        self.calls.append(("ingest", kwargs))
        return self.document

    def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return [self.result]

    def answer(self, **kwargs):
        self.calls.append(("answer", kwargs))
        return RagAnswer(answer="Milvus setup [Source 1]", sources=[self.result])


def test_rag_tool_ingest_dispatches_to_manager() -> None:
    manager = FakeRagManager()
    tool = RagTool(rag_manager=manager)

    result = tool.execute("ingest", source="/tmp/guide.md", source_type="file")

    assert "RAG 文档已摄取" in result
    assert "doc12345" in result
    assert manager.calls[0] == (
        "ingest",
        {"source": "/tmp/guide.md", "source_type": "file", "metadata": {}},
    )


def test_rag_tool_search_formats_sources() -> None:
    tool = RagTool(rag_manager=FakeRagManager())

    result = tool.execute("search", query="Milvus", top_k=1)

    assert "找到 1 个相关片段" in result
    assert "guide.md" in result
    assert "Milvus setup" in result


def test_rag_tool_answer_formats_answer_and_sources() -> None:
    tool = RagTool(rag_manager=FakeRagManager())

    result = tool.execute("answer", query="Milvus?", top_k=1)

    assert "Milvus setup [Source 1]" in result
    assert "来源" in result
    assert "guide.md" in result
