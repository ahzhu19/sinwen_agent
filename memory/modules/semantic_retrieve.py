"""SemanticMemory retrieve：向量 + 图 RRF 融合。"""

from __future__ import annotations

from typing import Any

from ..graph_relevance import build_ranks, reciprocal_rank_fusion
from .base import MemoryRecord


def retrieve_with_rrf(
    *,
    config: Any,
    store: Any,
    vectors: Any,
    embeddings: Any,
    concept_extractor: Any,
    user_id: str,
    query: str,
    limit: int = 5,
    session_id: str | None = None,
) -> list[MemoryRecord]:
    query_vector = embeddings.embed(query)
    search_limit = max(limit * 3, limit)
    hits = vectors.search(
        query_vector=query_vector,
        user_id=user_id,
        limit=search_limit,
        session_id=session_id,
    )
    vector_scores = {hit.memory_id: hit.score for hit in hits}

    if hasattr(store, "list_pending_embedding"):
        pending_facts = store.list_pending_embedding(
            user_id,
            session_id=session_id,
            limit=config.semantic_read_your_writes_limit,
        )
        for fact in pending_facts:
            vector_scores.setdefault(fact.id, 0.55)

    query_concepts = concept_extractor.extract(query, {})
    if hasattr(store, "compute_graph_relevance"):
        graph_scores = store.compute_graph_relevance(
            user_id,
            query_concepts,
            max_hops=config.semantic_graph_max_hops,
            hop_decay=config.semantic_graph_hop_decay,
            relation_weights=config.semantic_graph_relation_weights,
            session_id=session_id,
        )
    elif hasattr(store, "expand_graph_candidates"):
        graph_scores = store.expand_graph_candidates(
            user_id,
            query_concepts,
            max_hops=config.semantic_graph_max_hops,
            hop_decay=config.semantic_graph_hop_decay,
            limit=config.semantic_graph_expansion_limit,
            session_id=session_id,
        )
    else:
        graph_scores = {}

    expansion_scores = dict(
        sorted(graph_scores.items(), key=lambda item: item[1], reverse=True)[
            : config.semantic_graph_expansion_limit
        ]
    )

    candidate_ids = list(vector_scores.keys())
    for memory_id in expansion_scores:
        if memory_id not in candidate_ids:
            candidate_ids.append(memory_id)

    if not candidate_ids:
        return []

    facts = store.get_many(candidate_ids)
    vector_ranks = build_ranks(vector_scores) if vector_scores else {}
    graph_ranks = build_ranks(graph_scores) if graph_scores else {}
    rrf_scores = reciprocal_rank_fusion(
        {"vector": vector_ranks, "graph": graph_ranks},
        k=config.semantic_rrf_k,
    )

    scored: list[tuple[float, MemoryRecord]] = []
    for fact in facts:
        record = _semantic_fact_to_record(fact)
        importance_weight = 0.8 + (record.importance * 0.4)
        final_score = rrf_scores.get(fact.id, 0.0) * importance_weight
        scored.append((final_score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored[:limit]]


def _semantic_fact_to_record(fact: Any) -> MemoryRecord:
    metadata = dict(fact.metadata)
    metadata.setdefault("concepts", list(fact.concepts))
    return MemoryRecord(
        id=fact.id,
        content=fact.content,
        memory_type="semantic",
        importance=fact.importance,
        metadata=metadata,
    )
