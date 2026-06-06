"""Milvus 向量存储适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pymilvus import MilvusClient

from ..config import MemoryConfig
from ..milvus_guard import validate_collection_dimension


@dataclass(frozen=True)
class VectorSearchHit:
    memory_id: str
    score: float
    user_id: str | None = None
    session_id: str | None = None


class MilvusVectorStore(Protocol):
    @property
    def collection_name(self) -> str:
        ...

    def ensure_collection(self, vector_size: int) -> None:
        ...

    def upsert(
        self,
        memory_id: str,
        vector: list[float],
        user_id: str,
        session_id: str | None,
    ) -> None:
        ...

    def search(
        self,
        query_vector: list[float],
        user_id: str,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[VectorSearchHit]:
        ...

    def delete(self, memory_id: str) -> None:
        ...


class MilvusEpisodicVectorStore:
    """Milvus 向量存储实现。"""

    def __init__(
        self,
        uri: str,
        collection_name: str,
        metric_type: str = "COSINE",
        timeout: int = 30,
    ) -> None:
        self._uri = uri
        self._collection_name = collection_name
        self._metric_type = metric_type
        self._timeout = timeout
        self._client: MilvusClient | None = None
        self._collection_ready = False

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri, timeout=self._timeout)
        return self._client

    def ensure_collection(self, vector_size: int) -> None:
        client = self._get_client()
        validate_collection_dimension(client, self._collection_name, vector_size)
        if self._collection_ready:
            return
        if not client.has_collection(self._collection_name):
            client.create_collection(
                collection_name=self._collection_name,
                dimension=vector_size,
                metric_type=self._metric_type,
                auto_id=False,
                id_type="string",
                max_length=64,
            )
        self._collection_ready = True

    def upsert(
        self,
        memory_id: str,
        vector: list[float],
        user_id: str,
        session_id: str | None,
    ) -> None:
        self.ensure_collection(len(vector))
        client = self._get_client()
        client.upsert(
            collection_name=self._collection_name,
            data=[
                {
                    "id": memory_id,
                    "vector": vector,
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "session_id": session_id or "",
                }
            ],
        )

    def search(
        self,
        query_vector: list[float],
        user_id: str,
        limit: int = 10,
        session_id: str | None = None,
    ) -> list[VectorSearchHit]:
        self.ensure_collection(len(query_vector))
        client = self._get_client()

        filter_expr = f'user_id == "{_escape_filter(user_id)}"'
        if session_id:
            filter_expr += f' and session_id == "{_escape_filter(session_id)}"'

        results = client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            filter=filter_expr,
            limit=limit,
            output_fields=["memory_id", "user_id", "session_id"],
        )

        hits: list[VectorSearchHit] = []
        if not results:
            return hits

        for item in results[0]:
            entity = item.get("entity", {})
            memory_id = entity.get("memory_id") or item.get("id")
            if memory_id is None:
                continue
            hits.append(
                VectorSearchHit(
                    memory_id=str(memory_id),
                    score=float(item.get("distance", item.get("score", 0.0))),
                    user_id=entity.get("user_id"),
                    session_id=entity.get("session_id") or None,
                )
            )
        return hits

    def delete(self, memory_id: str) -> None:
        if not self._collection_ready and not self._get_client().has_collection(
            self._collection_name
        ):
            return
        self._get_client().delete(
            collection_name=self._collection_name,
            ids=[memory_id],
        )


def create_vector_store(
    config: MemoryConfig,
    collection_name: str | None = None,
) -> MilvusEpisodicVectorStore:
    resolved = collection_name
    if resolved is None:
        resolved = config.episodic_milvus_collection()
    return MilvusEpisodicVectorStore(
        uri=config.milvus_uri,
        collection_name=resolved,
        metric_type=config.milvus_metric_type,
        timeout=config.milvus_timeout,
    )


def _escape_filter(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
