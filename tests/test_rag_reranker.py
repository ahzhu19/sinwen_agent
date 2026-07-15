"""RAG reranker 测试。"""

from __future__ import annotations

from typing import Any

from rag.chunker import MarkdownChunker
from rag.ingestion import RagIngestionService
from rag.models import RagChunk, RagDocument, RagSearchResult
from rag.reranker import (
    LLMReranker,
    NoneReranker,
    create_reranker,
)
from rag.retriever import RagRetriever
from rag.storage import InMemoryRagStore
from tests.rag_fakes import FakeConverter, FakeEmbeddingProvider, FakeLLM, FakeVectorStore


def _make_result(result_id: str, score: float, content: str = "") -> RagSearchResult:
    chunk = RagChunk(
        id=result_id,
        document_id="doc1",
        chunk_index=0,
        content=content or f"内容 {result_id}",
        heading_path=[],
        token_count=5,
    )
    document = RagDocument(
        id="doc1",
        source_uri="/tmp/test.md",
        source_type="file",
        title="test.md",
        mime_type="text/markdown",
        content_hash="hash",
        markdown="",
        status="indexed",
    )
    return RagSearchResult(chunk=chunk, document=document, score=score)


# ---------- reranker 单元测试 ----------

def test_none_reranker_preserves_order_and_truncates() -> None:
    reranker = NoneReranker()
    results = [
        _make_result("a", 0.9),
        _make_result("b", 0.7),
        _make_result("c", 0.5),
    ]

    output = reranker.rerank("query", results, top_k=2)

    assert [r.chunk.id for r in output] == ["a", "b"]


def test_none_reranker_empty_input() -> None:
    reranker = NoneReranker()
    assert reranker.rerank("query", [], top_k=5) == []


class ScriptedLLM:
    """按 chunk_id 返回预设分数的 LLM（批量模式）。"""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls = 0

    def invoke(self, messages: Any, temperature: float = 0, **kwargs: Any) -> str:
        import json

        self.calls += 1
        content = messages[1]["content"]
        result = []
        for chunk_id, score in self.scores.items():
            if f"内容 {chunk_id}" in content:
                result.append(score)
        return json.dumps(result) if result else json.dumps([0.5])


def test_llm_reranker_reorders_by_relevance() -> None:
    """LLM 给出更高分数的候选排到前面。"""
    llm = ScriptedLLM({"a": 0.3, "b": 0.95, "c": 0.1})
    reranker = LLMReranker(llm)
    results = [
        _make_result("a", 0.9),
        _make_result("b", 0.7),
        _make_result("c", 0.5),
    ]

    output = reranker.rerank("query", results, top_k=3)

    assert [r.chunk.id for r in output] == ["b", "a", "c"]
    assert llm.calls == 1  # batch mode: single LLM call


def test_llm_reranker_truncates_to_top_k() -> None:
    llm = ScriptedLLM({"a": 0.9, "b": 0.8, "c": 0.7})
    reranker = LLMReranker(llm)
    results = [_make_result("a", 0.5), _make_result("b", 0.5), _make_result("c", 0.5)]

    output = reranker.rerank("query", results, top_k=2)

    assert len(output) == 2
    assert [r.chunk.id for r in output] == ["a", "b"]


def test_llm_reranker_empty_input() -> None:
    llm = ScriptedLLM({})
    reranker = LLMReranker(llm)
    assert reranker.rerank("query", [], top_k=5) == []
    assert llm.calls == 0


def test_llm_reranker_falls_back_on_llm_exception() -> None:
    class FailingLLM:
        def invoke(self, messages: Any, temperature: float = 0, **kwargs: Any) -> str:
            raise RuntimeError("LLM 不可用")

    reranker = LLMReranker(FailingLLM())
    results = [_make_result("a", 0.9), _make_result("b", 0.5)]

    output = reranker.rerank("query", results, top_k=2)

    # LLM 失败时回退到原始向量分数
    assert [r.chunk.id for r in output] == ["a", "b"]


def test_llm_reranker_clamps_score_to_range() -> None:
    class OverflowLLM:
        def invoke(self, messages: Any, temperature: float = 0, **kwargs: Any) -> str:
            return "[1.8]"

    reranker = LLMReranker(OverflowLLM())
    results = [_make_result("a", 0.5)]

    output = reranker.rerank("query", results, top_k=1)

    assert len(output) == 1


def test_create_reranker_factory() -> None:
    assert isinstance(create_reranker("none"), NoneReranker)
    assert isinstance(create_reranker(None), NoneReranker)
    assert isinstance(create_reranker(False), NoneReranker)
    assert isinstance(create_reranker("llm", llm=FakeLLM()), LLMReranker)
    assert isinstance(create_reranker(True, llm=FakeLLM()), LLMReranker)


def test_create_reranker_llm_without_llm_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="llm"):
        create_reranker("llm")


def test_create_reranker_unsupported_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="不支持的 rerank"):
        create_reranker("cross_encoder")


# ---------- RagRetriever 集成测试 ----------

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
    ingestion.ingest("/tmp/milvus.md")
    return store, vector_store, embeddings


def test_retriever_without_rerank_unchanged() -> None:
    store, vector_store, embeddings = _ingest_fixture()
    retriever = RagRetriever(
        store=store, vector_store=vector_store, embedding_provider=embeddings
    )

    results = retriever.search("Milvus collections", top_k=5)

    assert len(results) >= 1
    assert all(isinstance(r, RagSearchResult) for r in results)


def test_retriever_with_llm_rerank_reorders() -> None:
    store, vector_store, embeddings = _ingest_fixture()
    llm = FakeLLM("0.9")
    retriever = RagRetriever(
        store=store,
        vector_store=vector_store,
        embedding_provider=embeddings,
        llm=llm,
        rerank_candidate_factor=2,
    )

    results = retriever.search("Milvus collections", top_k=2, rerank="llm")

    assert len(results) <= 2
    assert llm.messages  # LLM 被调用打分


def test_retriever_rerank_none_explicit() -> None:
    store, vector_store, embeddings = _ingest_fixture()
    retriever = RagRetriever(
        store=store, vector_store=vector_store, embedding_provider=embeddings
    )

    results = retriever.search("Milvus", top_k=5, rerank="none")

    assert len(results) >= 1


# ---------- Tool 层透传测试 ----------

def test_tool_search_passes_rerank(monkeypatch) -> None:
    """RagTool._search 透传 rerank 到 rag_manager.search。"""
    from tools.builtin.rag_tool import RagTool

    captured: dict[str, Any] = {}

    class CapturingManager:
        def search(self, **kwargs: Any) -> list:
            captured.update(kwargs)
            return []

        def ingest(self, **kwargs: Any) -> Any:
            ...

        def answer(self, **kwargs: Any) -> Any:
            ...

        def list_documents(self, **kwargs: Any) -> list:
            ...

        def delete(self, **kwargs: Any) -> None:
            ...

        def reindex(self, **kwargs: Any) -> Any:
            ...

        def stats(self) -> dict[str, Any]:
            ...

    tool = RagTool(rag_manager=CapturingManager())
    tool.execute("search", query="测试", rerank="llm")

    assert captured.get("rerank") == "llm"
