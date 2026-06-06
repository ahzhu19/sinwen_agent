"""SemanticMemory list/count/clear helpers (no Neo4j required)."""

from __future__ import annotations

from tests.concept_fakes import StubConceptExtractor
from tests.semantic_fakes import create_semantic_bundle, create_semantic_memory_with_outbox


def test_semantic_list_count_and_remove_all() -> None:
    bundle = create_semantic_bundle()
    memory, _ = create_semantic_memory_with_outbox(
        bundle,
        user_id="u_sem",
        concept_extractor=StubConceptExtractor(),
    )
    session_a = "sess_a"
    session_b = "sess_b"

    id_a = memory.add(
        "事实 A",
        importance=0.8,
        metadata={"session_id": session_a, "concepts": ["A"]},
    )
    memory.add(
        "事实 B",
        importance=0.7,
        metadata={"session_id": session_b, "concepts": ["B"]},
    )

    assert memory.count_for_user() == 2
    assert memory.count_for_user(session_a) == 1

    listed = memory.list_for_user(session_id=session_a, limit=10)
    assert len(listed) == 1
    assert listed[0].id == id_a
    assert listed[0].memory_type == "semantic"

    removed = memory.remove_all_for_user(session_id=session_a)
    assert removed == 1
    assert memory.count_for_user() == 1
    assert memory.count_for_user(session_b) == 1

    cleared_all = memory.remove_all_for_user()
    assert cleared_all == 1
    assert memory.count_for_user() == 0
