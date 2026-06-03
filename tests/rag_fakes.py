"""Fake RAG dependencies for tests."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from rag.converter import ConvertedDocument
from rag.models import RagChunk, RagDocument
from rag.vector_store import VectorHit


class FakeConverter:
    def __init__(self, markdown: str, mime_type: str | None = "text/markdown") -> None:
        self.markdown = markdown
        self.mime_type = mime_type
        self.calls: list[str] = []

    def convert(self, source: str) -> ConvertedDocument:
        self.calls.append(source)
        return ConvertedDocument(
            markdown=self.markdown,
            title=Path(source).name,
            mime_type=self.mime_type,
        )


class FakeEmbeddingProvider:
    def __init__(self, vector_size: int = 8) -> None:
        self._vector_size = vector_size
        self.calls: list[list[str]] = []

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [_hash_to_vector(text, self._vector_size) for text in texts]


class FakeLLM:
    def __init__(self, response: str = "答案 [Source 1]") -> None:
        self.response = response
        self.messages: list[Any] = []

    def invoke(self, messages: Any, temperature: float = 0, **kwargs: Any) -> str:
        _ = temperature, kwargs
        self.messages.append(messages)
        return self.response


def _hash_to_vector(text: str, size: int) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [((digest[index % len(digest)] / 255.0) * 2) - 1 for index in range(size)]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}

    def upsert_many(self, vectors: list[tuple[RagChunk, list[float], RagDocument]]) -> None:
        for chunk, vector, document in vectors:
            self.records[chunk.id] = {
                "vector": vector,
                "document_id": document.id,
                "source_uri": document.source_uri,
            }

    def search(self, query_vector: list[float], limit: int = 5) -> list[VectorHit]:
        hits: list[VectorHit] = []
        for chunk_id, record in self.records.items():
            hits.append(
                VectorHit(
                    chunk_id=chunk_id,
                    score=_cosine(query_vector, record["vector"]),
                    document_id=record["document_id"],
                    source_uri=record["source_uri"],
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:limit]

    def delete_document(self, document_id: str) -> None:
        for chunk_id, record in list(self.records.items()):
            if record["document_id"] == document_id:
                self.records.pop(chunk_id)


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
