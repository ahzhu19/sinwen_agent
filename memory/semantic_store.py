"""Neo4j 语义记忆图存储。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from neo4j import GraphDatabase

from .config import MemoryConfig


@dataclass(frozen=True)
class SemanticFact:
    id: str
    user_id: str
    content: str
    importance: float
    concepts: list[str]
    metadata: dict[str, Any]


class SemanticMemoryStore(Protocol):
    def ensure_schema(self) -> None:
        ...

    def upsert_memory(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        concepts: list[str],
    ) -> SemanticFact:
        ...

    def get_many(self, memory_ids: list[str]) -> list[SemanticFact]:
        ...

    def score_related_memories(
        self,
        user_id: str,
        query_concepts: list[str],
        memory_ids: list[str],
    ) -> dict[str, float]:
        ...

    def delete(self, memory_id: str) -> None:
        ...


class Neo4jSemanticMemoryStore:
    """Neo4j 图存储实现。"""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        self._schema_ready = False

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        statements = [
            "CREATE CONSTRAINT semantic_memory_id IF NOT EXISTS FOR (m:SemanticMemory) REQUIRE m.id IS UNIQUE",
            "CREATE INDEX semantic_memory_user IF NOT EXISTS FOR (m:SemanticMemory) ON (m.user_id)",
            "CREATE INDEX semantic_concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
        ]
        with self._driver.session(database=self._database) as session:
            for statement in statements:
                session.run(statement)
        self._schema_ready = True

    def upsert_memory(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        concepts: list[str],
    ) -> SemanticFact:
        self.ensure_schema()
        normalized_concepts = _normalize_concepts(concepts)
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                MERGE (m:SemanticMemory {id: $memory_id})
                SET m.user_id = $user_id,
                    m.content = $content,
                    m.importance = $importance,
                    m.metadata = $metadata,
                    m.concepts = $concepts,
                    m.updated_at = timestamp()
                MERGE (u)-[:HAS_SEMANTIC_MEMORY]->(m)
                WITH m
                MATCH (m)-[old:MENTIONS]->(:Concept)
                DELETE old
                WITH m
                UNWIND $concepts AS concept
                MERGE (c:Concept {name: concept})
                MERGE (m)-[:MENTIONS]->(c)
                """,
                {
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "content": content,
                    "importance": importance,
                    "metadata": metadata,
                    "concepts": normalized_concepts,
                },
            )
        return SemanticFact(
            id=memory_id,
            user_id=user_id,
            content=content,
            importance=importance,
            concepts=normalized_concepts,
            metadata=dict(metadata),
        )

    def get_many(self, memory_ids: list[str]) -> list[SemanticFact]:
        if not memory_ids:
            return []
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (m:SemanticMemory)
                WHERE m.id IN $memory_ids
                RETURN m.id AS id,
                       m.user_id AS user_id,
                       m.content AS content,
                       m.importance AS importance,
                       m.metadata AS metadata,
                       m.concepts AS concepts
                """,
                {"memory_ids": memory_ids},
            ).data()
        order = {memory_id: index for index, memory_id in enumerate(memory_ids)}
        facts = [_row_to_fact(row) for row in rows]
        facts.sort(key=lambda fact: order.get(fact.id, len(memory_ids)))
        return facts

    def score_related_memories(
        self,
        user_id: str,
        query_concepts: list[str],
        memory_ids: list[str],
    ) -> dict[str, float]:
        concepts = _normalize_concepts(query_concepts)
        if not concepts or not memory_ids:
            return {memory_id: 0.0 for memory_id in memory_ids}
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (m:SemanticMemory)-[:MENTIONS]->(c:Concept)
                WHERE m.user_id = $user_id AND m.id IN $memory_ids AND c.name IN $concepts
                RETURN m.id AS id, count(DISTINCT c.name) AS matches
                """,
                {
                    "user_id": user_id,
                    "memory_ids": memory_ids,
                    "concepts": concepts,
                },
            ).data()
        scores = {memory_id: 0.0 for memory_id in memory_ids}
        denominator = len(concepts)
        for row in rows:
            scores[str(row["id"])] = min(1.0, float(row["matches"]) / denominator)
        return scores

    def delete(self, memory_id: str) -> None:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (m:SemanticMemory {id: $memory_id})
                DETACH DELETE m
                """,
                {"memory_id": memory_id},
            )


def create_semantic_store(config: MemoryConfig) -> Neo4jSemanticMemoryStore:
    if not config.neo4j_uri or not config.neo4j_username or not config.neo4j_password:
        raise ValueError("未配置 Neo4j 连接信息，无法启用语义记忆")
    return Neo4jSemanticMemoryStore(
        uri=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.neo4j_database,
    )


def _normalize_concepts(concepts: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for concept in concepts:
        value = str(concept).strip()
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def _row_to_fact(row: dict[str, Any]) -> SemanticFact:
    return SemanticFact(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        content=str(row["content"]),
        importance=float(row["importance"]),
        concepts=list(row.get("concepts") or []),
        metadata=dict(row.get("metadata") or {}),
    )
