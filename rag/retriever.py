"""RAG retrieval."""

from __future__ import annotations

from typing import Any

from .models import RagSearchResult
from .query_strategy import QueryStrategy, create_query_strategy
from .reranker import NoneReranker, Reranker, create_reranker
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
        reranker: Reranker | None = None,
        rerank_candidate_factor: int = 3,
    ) -> None:
        self._store = store
        self._vector_store = vector_store
        self._embeddings = embedding_provider
        self._query_strategy = query_strategy or create_query_strategy("direct")
        self._llm = llm
        self._reranker = reranker or NoneReranker()
        self._rerank_candidate_factor = max(1, rerank_candidate_factor)

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "direct",
        rerank: str | bool | None = None,
    ) -> list[RagSearchResult]:
        if strategy == "direct" and self._query_strategy is not None:
            query_strategy = self._query_strategy
        else:
            query_strategy = create_query_strategy(strategy, self._llm)

        active_reranker = self._reranker
        if rerank is not None:
            active_reranker = create_reranker(rerank, self._llm)

        candidate_limit = top_k
        if not isinstance(active_reranker, NoneReranker):
            candidate_limit = top_k * self._rerank_candidate_factor

        hits_by_id: dict[str, float] = {}
        for search_query in query_strategy.build_queries(query):
            query_vector = self._embeddings.embed(search_query)
            for hit in self._vector_store.search(query_vector, limit=candidate_limit):
                current = hits_by_id.get(hit.chunk_id)
                if current is None or hit.score > current:
                    hits_by_id[hit.chunk_id] = hit.score

        if not hits_by_id:
            return []

        sorted_hits = sorted(hits_by_id.items(), key=lambda item: item[1], reverse=True)
        chunk_ids = [chunk_id for chunk_id, _ in sorted_hits]
        chunks = self._store.get_chunks(chunk_ids)
        chunks_by_id = {chunk.id: chunk for chunk in chunks}

        unique_doc_ids = list({chunk.document_id for chunk in chunks})
        documents = self._store.get_documents(unique_doc_ids)
        documents_by_id = {doc.id: doc for doc in documents}

        results: list[RagSearchResult] = []
        for chunk_id, score in sorted_hits:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            document = documents_by_id.get(chunk.document_id)
            if document is None:
                continue
            results.append(RagSearchResult(chunk=chunk, document=document, score=score))
        return active_reranker.rerank(query, results, top_k)
