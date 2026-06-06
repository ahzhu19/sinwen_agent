"""语义记忆图扩展检索测试。"""

from __future__ import annotations

from memory.config import MemoryConfig
from tests.concept_fakes import StubConceptExtractor
from tests.semantic_fakes import create_semantic_bundle, create_semantic_memory_with_outbox


def test_semantic_retrieve_merges_graph_expansion_candidates() -> None:
    bundle = create_semantic_bundle()
    memory, _ = create_semantic_memory_with_outbox(
        bundle,
        concept_extractor=StubConceptExtractor(
            query_concepts=["Python", "机器学习"],
        ),
        config=MemoryConfig(
            semantic_graph_max_hops=2,
            semantic_graph_expansion_limit=10,
        ),
    )

    direct_id = memory.add(
        "Python 机器学习偏好",
        0.6,
        {"concepts": ["Python", "机器学习"]},
    )
    neighbor_id = memory.add(
        "深度学习框架 TensorFlow",
        0.9,
        {"concepts": ["深度学习", "TensorFlow"]},
    )
    bundle.store.graph_scores[direct_id] = 1.0
    bundle.store.graph_scores[neighbor_id] = 0.0
    bundle.store.expanded_scores[neighbor_id] = 0.85
    bundle.vectors.records.pop(neighbor_id, None)

    results = memory.retrieve("Python 机器学习", limit=3)

    assert any(record.id == neighbor_id for record in results)
    assert any(record.id == direct_id for record in results)
