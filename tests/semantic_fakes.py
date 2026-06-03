"""语义记忆测试用 fake 后端。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory.storage.neo4j_store import SemanticFact
from tests.episodic_fakes import FakeEmbeddingProvider, FakeVectorStore


class FakeSemanticStore:
    def __init__(self) -> None:
        self.facts: dict[str, SemanticFact] = {}
        self.graph_scores: dict[str, float] = {}

    def ensure_schema(self) -> None:
        return None

    def upsert_memory(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        concepts: list[str],
    ) -> SemanticFact:
        fact = SemanticFact(
            id=memory_id,
            user_id=user_id,
            content=content,
            importance=importance,
            concepts=concepts,
            metadata=dict(metadata),
        )
        self.facts[memory_id] = fact
        return fact

    def get_many(self, memory_ids: list[str]) -> list[SemanticFact]:
        return [self.facts[memory_id] for memory_id in memory_ids if memory_id in self.facts]

    def score_related_memories(
        self,
        user_id: str,
        query_concepts: list[str],
        memory_ids: list[str],
    ) -> dict[str, float]:
        _ = user_id
        _ = query_concepts
        return {
            memory_id: self.graph_scores.get(memory_id, 0.0)
            for memory_id in memory_ids
        }

    def delete(self, memory_id: str) -> None:
        self.facts.pop(memory_id, None)


@dataclass
class SemanticFakeBundle:
    store: FakeSemanticStore
    vectors: FakeVectorStore
    embeddings: FakeEmbeddingProvider


def create_semantic_bundle(vector_size: int = 8) -> SemanticFakeBundle:
    return SemanticFakeBundle(
        store=FakeSemanticStore(),
        vectors=FakeVectorStore(),
        embeddings=FakeEmbeddingProvider(vector_size=vector_size),
    )
