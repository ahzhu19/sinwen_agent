"""RAG vector store adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pymilvus import MilvusClient

from memory.milvus_guard import validate_collection_dimension

from .models import RagChunk, RagDocument


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float
    document_id: str
    source_uri: str


class RagVectorStore(Protocol):
    def upsert_many(self, vectors: list[tuple[RagChunk, list[float], RagDocument]]) -> None:
        ...

    def search(self, query_vector: list[float], limit: int = 5) -> list[VectorHit]:
        ...

    def delete_document(self, document_id: str) -> None:
        ...


class MilvusRagVectorStore:
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

    def _get_client(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._uri, timeout=self._timeout)
        return self._client

    def ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return
        client = self._get_client()
        validate_collection_dimension(client, self._collection_name, vector_size)
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

    def upsert_many(self, vectors: list[tuple[RagChunk, list[float], RagDocument]]) -> None:
        if not vectors:
            return
        self.ensure_collection(len(vectors[0][1]))
        data = [
            {
                "id": chunk.id,
                "vector": vector,
                "chunk_id": chunk.id,
                "document_id": document.id,
                "source_uri": document.source_uri,
            }
            for chunk, vector, document in vectors
        ]
        self._get_client().upsert(collection_name=self._collection_name, data=data)

    def search(self, query_vector: list[float], limit: int = 5) -> list[VectorHit]:
        self.ensure_collection(len(query_vector))
        results = self._get_client().search(
            collection_name=self._collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["chunk_id", "document_id", "source_uri"],
        )
        hits: list[VectorHit] = []
        if not results:
            return hits
        for item in results[0]:
            entity = item.get("entity", {})
            chunk_id = entity.get("chunk_id") or item.get("id")
            document_id = entity.get("document_id")
            source_uri = entity.get("source_uri")
            if chunk_id is None or document_id is None or source_uri is None:
                continue
            hits.append(
                VectorHit(
                    chunk_id=str(chunk_id),
                    score=float(item.get("distance", item.get("score", 0.0))),
                    document_id=str(document_id),
                    source_uri=str(source_uri),
                )
            )
        return hits

    def delete_document(self, document_id: str) -> None:
        if not self._collection_ready and not self._get_client().has_collection(
            self._collection_name
        ):
            return
        escaped = document_id.replace("\\", "\\\\").replace('"', '\\"')
        self._get_client().delete(
            collection_name=self._collection_name,
            filter=f'document_id == "{escaped}"',
        )
