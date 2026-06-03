"""Semantic store serialization tests."""

from __future__ import annotations

from memory.storage.neo4j_store import _metadata_to_property, _row_to_fact


def test_semantic_store_serializes_nested_metadata_for_neo4j_properties() -> None:
    metadata = {
        "session_id": "session_1",
        "source": "chat",
        "concepts": ["语义记忆", "Neo4j"],
        "extra": {"nested": True},
    }

    metadata_json = _metadata_to_property(metadata)
    fact = _row_to_fact(
        {
            "id": "memory_1",
            "user_id": "user123",
            "content": "语义记忆使用 Neo4j",
            "importance": 0.8,
            "concepts": ["语义记忆", "Neo4j"],
            "metadata_json": metadata_json,
        }
    )

    assert fact.metadata == metadata
