"""RAG retrieval."""

from __future__ import annotations

from typing import Any

from .models import RagSearchResult
from .query_strategy import QueryStrategy, create_query_strategy
from .storage import RagStore
from .vector_store import RagVectorStore


class RagRetriever:
    def __init__(
        self,
        store: RagStore,
        vector_store: RagVectorStore,
        embedding_provider: Any,
        query_strategy: QueryStrategy | None = None,
        llm: Any | None = None,
    ) -> None:
        self._store = store
        self._vector_store = vector_store
        self._embeddings = embedding_provider
        self._query_strategy = query_strategy or create_query_strategy("direct")
        self._llm = llm

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "direct",
    ) -> list[RagSearchResult]:
        if strategy == "direct" and self._query_strategy is not None:
            query_strategy = self._query_strategy
        else:
            query_strategy = create_query_strategy(strategy, self._llm)

        hits_by_id: dict[str, float] = {}
        for search_query in query_strategy.build_queries(query):
            query_vector = self._embeddings.embed(search_query)
            for hit in self._vector_store.search(query_vector, limit=top_k):
                current = hits_by_id.get(hit.chunk_id)
                if current is None or hit.score > current:
                    hits_by_id[hit.chunk_id] = hit.score

        if not hits_by_id:
            return []

        sorted_hits = sorted(hits_by_id.items(), key=lambda item: item[1], reverse=True)[
            :top_k
        ]
        chunk_ids = [chunk_id for chunk_id, _ in sorted_hits]
        chunks = self._store.get_chunks(chunk_ids)
        chunks_by_id = {chunk.id: chunk for chunk in chunks}

        results: list[RagSearchResult] = []
        for chunk_id, score in sorted_hits:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            document = self._store.get_document(chunk.document_id)
            results.append(RagSearchResult(chunk=chunk, document=document, score=score))
        return results
