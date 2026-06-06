"""图检索 RRF 与 collection 命名测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory.collection_names import resolve_collection_name, versioned_collection_name
from memory.forget_policy import should_forget_record
from memory.graph_relevance import build_ranks, reciprocal_rank_fusion


def test_versioned_collection_name() -> None:
    name = versioned_collection_name("hello_agents_semantic_vectors", "text-embedding-v3", 1024)
    assert name == "hello_agents_semantic_vectors_text_embedding_v3_1024"


def test_resolve_collection_name_can_disable_versioning() -> None:
    assert resolve_collection_name(
        "base",
        embed_model="text-embedding-v3",
        vector_size=1024,
        use_versioned=False,
    ) == "base"


def test_rrf_merges_vector_and_graph_tracks() -> None:
    vector_ranks = build_ranks({"a": 0.9, "b": 0.5})
    graph_ranks = build_ranks({"b": 0.95, "c": 0.8})
    fused = reciprocal_rank_fusion({"vector": vector_ranks, "graph": graph_ranks}, k=60)
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]
    assert "c" in fused


def test_should_forget_importance_ttl() -> None:
    old = datetime.now(timezone.utc) - timedelta(days=40)
    recent = datetime.now(timezone.utc) - timedelta(days=1)
    assert should_forget_record(
        importance=0.1,
        importance_threshold=0.2,
        strategy="importance_ttl",
        occurred_at=old,
        older_than_days=30,
    )
    assert not should_forget_record(
        importance=0.1,
        importance_threshold=0.2,
        strategy="importance_ttl",
        occurred_at=recent,
        older_than_days=30,
    )
