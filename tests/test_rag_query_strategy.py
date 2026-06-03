"""RAG query strategy tests."""

from rag.query_strategy import (
    DirectQueryStrategy,
    HyDEQueryStrategy,
    MultiQueryStrategy,
    create_query_strategy,
)
from tests.rag_fakes import FakeLLM


def test_direct_strategy_returns_single_query() -> None:
    strategy = DirectQueryStrategy()
    assert strategy.build_queries("  Milvus  ") == ["Milvus"]


def test_hyde_strategy_adds_hypothetical_document() -> None:
    llm = FakeLLM("假设段落：向量数据库用 collection 组织数据。")
    strategy = HyDEQueryStrategy(llm)
    queries = strategy.build_queries("Milvus 是什么？")
    assert queries[0] == "Milvus 是什么？"
    assert "collection" in queries[1]


def test_multi_query_strategy_parses_subqueries() -> None:
    llm = FakeLLM('["Milvus 架构", "Milvus 检索 API"]')
    strategy = MultiQueryStrategy(llm)
    queries = strategy.build_queries("如何使用 Milvus？")
    assert "如何使用 Milvus？" in queries
    assert "Milvus 架构" in queries


def test_create_query_strategy_rejects_unknown() -> None:
    try:
        create_query_strategy("unknown")
    except ValueError as exc:
        assert "不支持的查询策略" in str(exc)
    else:
        raise AssertionError("expected ValueError")
