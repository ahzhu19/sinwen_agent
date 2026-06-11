"""Neo4j 语义记忆图存储。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from neo4j import GraphDatabase

from ..config import MemoryConfig
from .semantic_outbox_types import SemanticMemorySyncState, SemanticOutboxEvent


@dataclass(frozen=True)
class SemanticFact:
    id: str
    user_id: str
    content: str
    importance: float
    concepts: list[str]
    metadata: dict[str, Any]
    version: int = 1
    embedding_version: int = 0
    embedding_status: str = "done"
    embedding_model: str = ""
    deleted: bool = False


class SemanticMemoryStore(Protocol):
    def ensure_schema(self) -> None:
        ...

    def get_many(self, memory_ids: list[str]) -> list[SemanticFact]:
        ...

    def compute_graph_relevance(
        self,
        user_id: str,
        query_concepts: list[str],
        *,
        max_hops: int = 2,
        hop_decay: float = 0.65,
        relation_weights: dict[str, float] | None = None,
        session_id: str | None = None,
    ) -> dict[str, float]:
        ...

    def list_by_user(
        self,
        user_id: str,
        limit: int = 10_000,
        session_id: str | None = None,
    ) -> list[SemanticFact]:
        ...

    def count_by_user(self, user_id: str, session_id: str | None = None) -> int:
        ...

    def write_memory_with_outbox(
        self,
        *,
        user_id: str,
        memory_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        concepts: list[str],
        operation: str,
        embedding_model: str,
        collection_name: str,
        max_attempts: int = 5,
    ) -> SemanticFact:
        ...

    def delete_memory_with_outbox(
        self,
        *,
        memory_id: str,
        user_id: str,
        embedding_model: str,
        collection_name: str,
        max_attempts: int = 5,
    ) -> None:
        ...

    def claim_pending_outbox_events(self, *, batch_size: int = 20) -> list[SemanticOutboxEvent]:
        ...

    def mark_outbox_done(self, event_id: str) -> None:
        ...

    def mark_outbox_failed(self, event_id: str, error: str, *, max_attempts: int) -> None:
        ...

    def mark_outbox_superseded(self, event_id: str) -> None:
        ...

    def get_memory_sync_state(self, memory_id: str) -> SemanticMemorySyncState | None:
        ...

    def update_embedding_sync_state(
        self,
        memory_id: str,
        *,
        embedding_version: int,
        embedding_status: str,
        embedding_model: str,
    ) -> None:
        ...

    def list_pending_embedding(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[SemanticFact]:
        ...

    def pending_outbox_count(self) -> int:
        ...

    def outbox_status_counts(self) -> dict[str, int]:
        ...

    def reclaim_stale_processing_outbox(self, *, timeout_seconds: int) -> int:
        ...

    def replay_dead_outbox(self, *, batch_size: int = 20) -> int:
        ...

    def ensure_pending_outbox_events(
        self,
        *,
        batch_size: int = 20,
        max_attempts: int = 5,
        collection_name: str,
    ) -> int:
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
            "CREATE CONSTRAINT semantic_outbox_event_id IF NOT EXISTS FOR (e:SemanticOutboxEvent) REQUIRE e.id IS UNIQUE",
            "CREATE INDEX semantic_outbox_status IF NOT EXISTS FOR (e:SemanticOutboxEvent) ON (e.status)",
            "CREATE INDEX semantic_outbox_memory IF NOT EXISTS FOR (e:SemanticOutboxEvent) ON (e.memory_id)",
        ]
        with self._driver.session(database=self._database) as session:
            for statement in statements:
                session.run(statement)
        self._schema_ready = True

    def get_many(self, memory_ids: list[str]) -> list[SemanticFact]:
        if not memory_ids:
            return []
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (m:SemanticMemory)
                WHERE m.id IN $memory_ids AND coalesce(m.deleted, false) = false
                RETURN m.id AS id,
                       m.user_id AS user_id,
                       m.content AS content,
                       m.importance AS importance,
                       m.metadata_json AS metadata_json,
                       m.concepts AS concepts,
                       coalesce(m.version, 1) AS version,
                       coalesce(m.embedding_version, 0) AS embedding_version,
                       coalesce(m.embedding_status, 'done') AS embedding_status,
                       coalesce(m.embedding_model, '') AS embedding_model,
                       coalesce(m.deleted, false) AS deleted
                """,
                {"memory_ids": memory_ids},
            ).data()
        order = {memory_id: index for index, memory_id in enumerate(memory_ids)}
        facts = [_row_to_fact(row) for row in rows]
        facts.sort(key=lambda fact: order.get(fact.id, len(memory_ids)))
        return facts

    def compute_graph_relevance(
        self,
        user_id: str,
        query_concepts: list[str],
        *,
        max_hops: int = 2,
        hop_decay: float = 0.65,
        relation_weights: dict[str, float] | None = None,
        session_id: str | None = None,
    ) -> dict[str, float]:
        concepts = _normalize_concepts(query_concepts)
        if not concepts:
            return {}

        weights = relation_weights or {
            "RELATES_TO": 1.0,
            "CO_OCCURRENCE": 0.75,
        }
        relates_weight = float(weights.get("RELATES_TO", 1.0))
        cooc_weight = float(weights.get("CO_OCCURRENCE", 0.75))

        self.ensure_schema()
        scores: dict[str, float] = {}
        denominator = len(concepts)

        with self._driver.session(database=self._database) as session:
            hop1_rows = session.run(
                """
                MATCH (m:SemanticMemory {user_id: $user_id})-[:MENTIONS]->(c:Concept)
                WHERE c.name IN $concepts AND coalesce(m.deleted, false) = false
                RETURN m.id AS id, count(DISTINCT c.name) AS matches
                """,
                {"user_id": user_id, "concepts": concepts},
            ).data()

        for row in hop1_rows:
            memory_id = str(row["id"])
            hop_score = min(1.0, float(row["matches"]) / denominator)
            scores[memory_id] = max(scores.get(memory_id, 0.0), hop_score)

        if max_hops >= 2:
            hop2_weight = hop_decay**1
            with self._driver.session(database=self._database) as session:
                hop2_rows = session.run(
                    """
                    MATCH (qc:Concept)
                    WHERE qc.name IN $concepts
                    MATCH (qc)-[r:RELATES_TO]-(bridge:Concept)
                    MATCH (m:SemanticMemory {user_id: $user_id})-[:MENTIONS]->(bridge)
                    WHERE coalesce(m.deleted, false) = false
                    RETURN m.id AS id,
                           count(DISTINCT bridge.name) AS bridges,
                           avg(coalesce(r.weight, 1.0)) AS relation_weight
                    """,
                    {"user_id": user_id, "concepts": concepts},
                ).data()

                bridge_rows = session.run(
                    """
                    MATCH (m0:SemanticMemory {user_id: $user_id})-[:MENTIONS]->(qc:Concept)
                    WHERE qc.name IN $concepts AND coalesce(m0.deleted, false) = false
                    MATCH (m0)-[:MENTIONS]->(bridge:Concept)
                    MATCH (m1:SemanticMemory {user_id: $user_id})-[:MENTIONS]->(bridge)
                    WHERE m1.id <> m0.id AND coalesce(m1.deleted, false) = false
                    RETURN m1.id AS id, count(DISTINCT bridge.name) AS bridges
                    """,
                    {"user_id": user_id, "concepts": concepts},
                ).data()

            for row in hop2_rows:
                memory_id = str(row["id"])
                bridges = float(row.get("bridges") or 0)
                relation_weight = float(row.get("relation_weight") or 1.0)
                hop_score = (
                    min(1.0, bridges / max(1.0, denominator))
                    * relation_weight
                    * relates_weight
                    * hop2_weight
                )
                scores[memory_id] = max(scores.get(memory_id, 0.0), hop_score)

            for row in bridge_rows:
                memory_id = str(row["id"])
                bridges = float(row.get("bridges") or 0)
                hop_score = (
                    min(1.0, bridges / max(1.0, denominator))
                    * cooc_weight
                    * hop2_weight
                )
                scores[memory_id] = max(scores.get(memory_id, 0.0), hop_score)

        if session_id is not None:
            scores = self._filter_scores_by_session(user_id, scores, session_id)

        return scores

    def _filter_scores_by_session(
        self,
        user_id: str,
        scores: dict[str, float],
        session_id: str,
    ) -> dict[str, float]:
        if not scores:
            return scores
        facts = self.get_many(list(scores.keys()))
        allowed = {
            fact.id
            for fact in facts
            if fact.user_id == user_id and fact.metadata.get("session_id") == session_id
        }
        return {memory_id: value for memory_id, value in scores.items() if memory_id in allowed}

    def list_by_user(
        self,
        user_id: str,
        limit: int = 10_000,
        session_id: str | None = None,
    ) -> list[SemanticFact]:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (m:SemanticMemory {user_id: $user_id})
                WHERE coalesce(m.deleted, false) = false
                RETURN m.id AS id,
                       m.user_id AS user_id,
                       m.content AS content,
                       m.importance AS importance,
                       m.metadata_json AS metadata_json,
                       m.concepts AS concepts,
                       coalesce(m.version, 1) AS version,
                       coalesce(m.embedding_version, 0) AS embedding_version,
                       coalesce(m.embedding_status, 'done') AS embedding_status,
                       coalesce(m.embedding_model, '') AS embedding_model,
                       coalesce(m.deleted, false) AS deleted
                ORDER BY m.updated_at DESC
                LIMIT $limit
                """,
                {"user_id": user_id, "limit": limit},
            ).data()
        facts = [_row_to_fact(row) for row in rows]
        if session_id is None:
            return facts
        return [fact for fact in facts if fact.metadata.get("session_id") == session_id]

    def write_memory_with_outbox(
        self,
        *,
        user_id: str,
        memory_id: str,
        content: str,
        importance: float,
        metadata: dict[str, Any],
        concepts: list[str],
        operation: str,
        embedding_model: str,
        collection_name: str,
        max_attempts: int = 5,
    ) -> SemanticFact:
        if operation not in {"create", "update"}:
            raise ValueError(f"不支持的 outbox operation: {operation}")
        self.ensure_schema()
        normalized_concepts = _normalize_concepts(concepts)
        metadata_json = _metadata_to_property(metadata)
        session_id = metadata.get("session_id")
        session_value = session_id if isinstance(session_id, str) else None
        event_id = str(uuid4())

        with self._driver.session(database=self._database) as session:
            session.execute_write(
                _write_memory_with_outbox_tx,
                {
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "content": content,
                    "importance": importance,
                    "metadata_json": metadata_json,
                    "concepts": normalized_concepts,
                    "operation": operation,
                    "embedding_model": embedding_model,
                    "collection_name": collection_name,
                    "max_attempts": max_attempts,
                    "event_id": event_id,
                    "session_id": session_value,
                },
            )

        state = self.get_memory_sync_state(memory_id)
        if state is None:
            raise RuntimeError(f"写入后未找到语义记忆: {memory_id}")
        return SemanticFact(
            id=state.id,
            user_id=state.user_id,
            content=state.content,
            importance=state.importance,
            concepts=list(state.concepts),
            metadata=dict(state.metadata),
            version=state.version,
            embedding_version=state.embedding_version,
            embedding_status=state.embedding_status,
            embedding_model=state.embedding_model,
            deleted=state.deleted,
        )

    def delete_memory_with_outbox(
        self,
        *,
        memory_id: str,
        user_id: str,
        embedding_model: str,
        collection_name: str,
        max_attempts: int = 5,
    ) -> None:
        self.ensure_schema()
        event_id = str(uuid4())
        with self._driver.session(database=self._database) as session:
            session.execute_write(
                _delete_memory_with_outbox_tx,
                {
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "embedding_model": embedding_model,
                    "collection_name": collection_name,
                    "max_attempts": max_attempts,
                    "event_id": event_id,
                },
            )

    def claim_pending_outbox_events(self, *, batch_size: int = 20) -> list[SemanticOutboxEvent]:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (e:SemanticOutboxEvent)
                WHERE e.status = 'pending' AND e.attempts < e.max_attempts
                WITH e ORDER BY e.created_at ASC LIMIT $batch_size
                SET e.status = 'processing',
                    e.attempts = coalesce(e.attempts, 0) + 1,
                    e.updated_at = timestamp()
                RETURN e.id AS event_id,
                       e.memory_id AS memory_id,
                       e.user_id AS user_id,
                       e.version AS version,
                       e.operation AS operation,
                       e.status AS status,
                       e.attempts AS attempts,
                       e.max_attempts AS max_attempts,
                       e.last_error AS last_error,
                       e.embedding_model AS embedding_model,
                       e.collection_name AS collection_name,
                       e.session_id AS session_id
                """,
                {"batch_size": batch_size},
            ).data()
        return [_row_to_outbox_event(row) for row in rows]

    def mark_outbox_done(self, event_id: str) -> None:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (e:SemanticOutboxEvent {id: $event_id})
                SET e.status = 'done', e.processed_at = timestamp(), e.last_error = null
                """,
                {"event_id": event_id},
            )

    def mark_outbox_failed(self, event_id: str, error: str, *, max_attempts: int) -> None:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (e:SemanticOutboxEvent {id: $event_id})
                SET e.last_error = $error,
                    e.updated_at = timestamp(),
                    e.status = CASE
                        WHEN e.attempts >= $max_attempts THEN 'dead'
                        ELSE 'pending'
                    END
                """,
                {"event_id": event_id, "error": error[:2000], "max_attempts": max_attempts},
            )

    def mark_outbox_superseded(self, event_id: str) -> None:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (e:SemanticOutboxEvent {id: $event_id})
                SET e.status = 'superseded', e.processed_at = timestamp(), e.last_error = null
                """,
                {"event_id": event_id},
            )

    def get_memory_sync_state(self, memory_id: str) -> SemanticMemorySyncState | None:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            row = session.run(
                """
                MATCH (m:SemanticMemory {id: $memory_id})
                RETURN m.id AS id,
                       m.user_id AS user_id,
                       m.content AS content,
                       m.importance AS importance,
                       m.metadata_json AS metadata_json,
                       m.concepts AS concepts,
                       coalesce(m.version, 1) AS version,
                       coalesce(m.embedding_version, 0) AS embedding_version,
                       coalesce(m.embedding_status, 'done') AS embedding_status,
                       coalesce(m.embedding_model, '') AS embedding_model,
                       coalesce(m.deleted, false) AS deleted
                """,
                {"memory_id": memory_id},
            ).single()
        if row is None:
            return None
        fact = _row_to_fact(dict(row))
        session_id = fact.metadata.get("session_id")
        session_value = session_id if isinstance(session_id, str) else None
        return SemanticMemorySyncState(
            id=fact.id,
            user_id=fact.user_id,
            content=fact.content,
            importance=fact.importance,
            concepts=list(fact.concepts),
            metadata=dict(fact.metadata),
            version=fact.version,
            embedding_version=fact.embedding_version,
            embedding_status=fact.embedding_status,
            embedding_model=fact.embedding_model,
            deleted=fact.deleted,
            session_id=session_value,
        )

    def update_embedding_sync_state(
        self,
        memory_id: str,
        *,
        embedding_version: int,
        embedding_status: str,
        embedding_model: str,
    ) -> None:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (m:SemanticMemory {id: $memory_id})
                SET m.embedding_version = $embedding_version,
                    m.embedding_status = $embedding_status,
                    m.embedding_model = $embedding_model,
                    m.updated_at = timestamp()
                """,
                {
                    "memory_id": memory_id,
                    "embedding_version": embedding_version,
                    "embedding_status": embedding_status,
                    "embedding_model": embedding_model,
                },
            )

    def list_pending_embedding(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[SemanticFact]:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (m:SemanticMemory {user_id: $user_id})
                WHERE m.embedding_status = 'pending'
                  AND coalesce(m.deleted, false) = false
                RETURN m.id AS id,
                       m.user_id AS user_id,
                       m.content AS content,
                       m.importance AS importance,
                       m.metadata_json AS metadata_json,
                       m.concepts AS concepts,
                       coalesce(m.version, 1) AS version,
                       coalesce(m.embedding_version, 0) AS embedding_version,
                       coalesce(m.embedding_status, 'done') AS embedding_status,
                       coalesce(m.embedding_model, '') AS embedding_model,
                       coalesce(m.deleted, false) AS deleted
                ORDER BY m.updated_at DESC
                LIMIT $limit
                """,
                {"user_id": user_id, "limit": limit},
            ).data()
        facts = [_row_to_fact(row) for row in rows]
        if session_id is None:
            return facts
        return [fact for fact in facts if fact.metadata.get("session_id") == session_id]

    def pending_outbox_count(self) -> int:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            row = session.run(
                """
                MATCH (e:SemanticOutboxEvent)
                WHERE e.status IN ['pending', 'processing']
                RETURN count(e) AS total
                """
            ).single()
        return int(row["total"]) if row else 0

    def outbox_status_counts(self) -> dict[str, int]:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (e:SemanticOutboxEvent)
                WHERE e.status IN ['pending', 'processing', 'dead']
                RETURN e.status AS status, count(e) AS total
                """
            ).data()

        counts = {"pending": 0, "processing": 0, "dead": 0}
        for row in rows:
            status = str(row["status"])
            if status in counts:
                counts[status] = int(row["total"])
        return counts

    def reclaim_stale_processing_outbox(self, *, timeout_seconds: int) -> int:
        self.ensure_schema()
        timeout_ms = timeout_seconds * 1000
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (e:SemanticOutboxEvent)
                WHERE e.status = 'processing'
                  AND e.updated_at < timestamp() - $timeout_ms
                SET e.status = 'pending',
                    e.last_error = coalesce(e.last_error, '') + ' [reclaimed stale processing]'
                RETURN count(e) AS total
                """,
                {"timeout_ms": timeout_ms},
            ).single()
        return int(result["total"]) if result else 0

    def replay_dead_outbox(self, *, batch_size: int = 20) -> int:
        self.ensure_schema()
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (e:SemanticOutboxEvent)
                WHERE e.status = 'dead'
                WITH e ORDER BY e.updated_at ASC LIMIT $batch_size
                SET e.status = 'pending',
                    e.attempts = 0,
                    e.last_error = null,
                    e.updated_at = timestamp()
                RETURN count(e) AS total
                """,
                {"batch_size": batch_size},
            ).single()
        return int(result["total"]) if result else 0

    def ensure_pending_outbox_events(
        self,
        *,
        batch_size: int = 20,
        max_attempts: int = 5,
        collection_name: str,
    ) -> int:
        """为 embedding 未同步且无活跃 outbox 的语义记忆补建 pending 事件。"""
        self.ensure_schema()
        created = 0
        with self._driver.session(database=self._database) as session:
            rows = session.run(
                """
                MATCH (m:SemanticMemory)
                WHERE coalesce(m.deleted, false) = false
                  AND (
                    m.embedding_status = 'pending'
                    OR coalesce(m.embedding_version, 0) < coalesce(m.version, 1)
                  )
                OPTIONAL MATCH (e:SemanticOutboxEvent {memory_id: m.id})
                WHERE e.status IN ['pending', 'processing']
                  AND e.version = coalesce(m.version, 1)
                WITH m WHERE e IS NULL
                RETURN m.id AS memory_id,
                       m.user_id AS user_id,
                       coalesce(m.version, 1) AS version,
                       coalesce(m.embedding_model, '') AS embedding_model,
                       m.metadata_json AS metadata_json
                ORDER BY m.updated_at ASC
                LIMIT $batch_size
                """,
                {"batch_size": batch_size},
            ).data()

            for row in rows:
                metadata_json = row.get("metadata_json") or "{}"
                if isinstance(metadata_json, str):
                    metadata = json.loads(metadata_json) if metadata_json else {}
                else:
                    metadata = {}
                session_id = metadata.get("session_id")
                session_value = session_id if isinstance(session_id, str) else None
                event_id = str(uuid4())
                session.run(
                    """
                    MATCH (m:SemanticMemory {id: $memory_id})
                    CREATE (e:SemanticOutboxEvent {
                        id: $event_id,
                        memory_id: $memory_id,
                        user_id: $user_id,
                        version: $version,
                        operation: 'update',
                        status: 'pending',
                        attempts: 0,
                        max_attempts: $max_attempts,
                        embedding_model: $embedding_model,
                        collection_name: $collection_name,
                        session_id: $session_id,
                        created_at: timestamp(),
                        updated_at: timestamp()
                    })
                    MERGE (m)-[:HAS_OUTBOX_EVENT]->(e)
                    """,
                    {
                        "event_id": event_id,
                        "memory_id": row["memory_id"],
                        "user_id": row["user_id"],
                        "version": int(row["version"]),
                        "max_attempts": max_attempts,
                        "embedding_model": row["embedding_model"],
                        "collection_name": collection_name,
                        "session_id": session_value,
                    },
                )
                created += 1
        return created

    def count_by_user(self, user_id: str, session_id: str | None = None) -> int:
        if session_id is None:
            self.ensure_schema()
            with self._driver.session(database=self._database) as session:
                row = session.run(
                    """
                    MATCH (m:SemanticMemory {user_id: $user_id})
                    WHERE coalesce(m.deleted, false) = false
                    RETURN count(m) AS total
                    """,
                    {"user_id": user_id},
                ).single()
            return int(row["total"]) if row else 0
        return len(self.list_by_user(user_id, limit=10_000, session_id=session_id))


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


def _metadata_to_property(metadata: dict[str, Any]) -> str:
    return json.dumps(metadata, ensure_ascii=False)


def _row_to_fact(row: dict[str, Any]) -> SemanticFact:
    metadata_json = row.get("metadata_json")
    if isinstance(metadata_json, str) and metadata_json:
        metadata = json.loads(metadata_json)
    else:
        metadata = {}

    return SemanticFact(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        content=str(row["content"]),
        importance=float(row["importance"]),
        concepts=list(row.get("concepts") or []),
        metadata=dict(metadata),
        version=int(row.get("version") or 1),
        embedding_version=int(row.get("embedding_version") or 0),
        embedding_status=str(row.get("embedding_status") or "done"),
        embedding_model=str(row.get("embedding_model") or ""),
        deleted=bool(row.get("deleted") or False),
    )


def _row_to_outbox_event(row: dict[str, Any]) -> SemanticOutboxEvent:
    session_id = row.get("session_id")
    session_value = session_id if isinstance(session_id, str) and session_id else None
    return SemanticOutboxEvent(
        event_id=str(row["event_id"]),
        memory_id=str(row["memory_id"]),
        user_id=str(row["user_id"]),
        version=int(row["version"]),
        operation=str(row["operation"]),
        status=str(row["status"]),
        attempts=int(row.get("attempts") or 0),
        max_attempts=int(row.get("max_attempts") or 5),
        last_error=row.get("last_error"),
        embedding_model=str(row.get("embedding_model") or ""),
        collection_name=str(row.get("collection_name") or ""),
        session_id=session_value,
    )


def _write_memory_with_outbox_tx(tx: Any, params: dict[str, Any]) -> None:
    tx.run(
        """
        MERGE (u:User {id: $user_id})
        MERGE (m:SemanticMemory {id: $memory_id})
        SET m.user_id = $user_id,
            m.content = $content,
            m.importance = $importance,
            m.metadata_json = $metadata_json,
            m.concepts = $concepts,
            m.deleted = false,
            m.version = CASE
                WHEN $operation = 'create' AND m.version IS NULL THEN 1
                WHEN $operation = 'create' THEN coalesce(m.version, 0) + 1
                ELSE coalesce(m.version, 0) + 1
            END,
            m.embedding_status = 'pending',
            m.updated_at = timestamp()
        MERGE (u)-[:HAS_SEMANTIC_MEMORY]->(m)
        WITH m
        MATCH (m)-[old:MENTIONS]->(:Concept)
        DELETE old
        WITH m
        UNWIND $concepts AS concept
        MERGE (c:Concept {name: concept})
        MERGE (m)-[:MENTIONS]->(c)
        WITH m, collect(DISTINCT c) AS concept_nodes
        UNWIND concept_nodes AS c1
        UNWIND concept_nodes AS c2
        WITH m, c1, c2 WHERE id(c1) < id(c2)
        MERGE (c1)-[:RELATES_TO {weight: 1.0}]->(c2)
        WITH m
        CREATE (e:SemanticOutboxEvent {
            id: $event_id,
            memory_id: $memory_id,
            user_id: $user_id,
            version: m.version,
            operation: $operation,
            status: 'pending',
            attempts: 0,
            max_attempts: $max_attempts,
            embedding_model: $embedding_model,
            collection_name: $collection_name,
            session_id: $session_id,
            created_at: timestamp(),
            updated_at: timestamp()
        })
        MERGE (m)-[:HAS_OUTBOX_EVENT]->(e)
        """,
        params,
    )


def _delete_memory_with_outbox_tx(tx: Any, params: dict[str, Any]) -> None:
    tx.run(
        """
        MATCH (m:SemanticMemory {id: $memory_id, user_id: $user_id})
        SET m.deleted = true,
            m.version = coalesce(m.version, 0) + 1,
            m.embedding_status = 'pending',
            m.updated_at = timestamp()
        WITH m
        CREATE (e:SemanticOutboxEvent {
            id: $event_id,
            memory_id: $memory_id,
            user_id: $user_id,
            version: m.version,
            operation: 'delete',
            status: 'pending',
            attempts: 0,
            max_attempts: $max_attempts,
            embedding_model: $embedding_model,
            collection_name: $collection_name,
            session_id: null,
            created_at: timestamp(),
            updated_at: timestamp()
        })
        MERGE (m)-[:HAS_OUTBOX_EVENT]->(e)
        """,
        params,
    )
