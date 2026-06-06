"""RAG configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class RagConfig:
    database_url: str | None = None
    milvus_uri: str = "http://localhost:19530"
    collection_name: str = "hello_agents_rag_chunks"
    vector_size: int = 1024
    metric_type: str = "COSINE"
    timeout: int = 30
    target_chunk_tokens: int = 500
    max_chunk_tokens: int = 800
    overlap_tokens: int = 80
    enable_rag_vector_outbox: bool = True
    rag_vector_outbox_max_attempts: int = 5
    rag_vector_outbox_processing_timeout_seconds: int = 900

    @classmethod
    def from_env(cls) -> RagConfig:
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = os.getenv("POSTGRES_PORT", "5432")
            db = os.getenv("POSTGRES_DB", "hello_agents")
            user = os.getenv("POSTGRES_USER", "hello_agents")
            password = os.getenv("POSTGRES_PASSWORD", "hello-agents-password")
            database_url = f"postgresql://{user}:{password}@{host}:{port}/{db}"

        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        return cls(
            database_url=database_url,
            milvus_uri=os.getenv("MILVUS_URI", f"http://{milvus_host}:{milvus_port}"),
            collection_name=os.getenv("MILVUS_RAG_COLLECTION", "hello_agents_rag_chunks"),
            vector_size=int(os.getenv("MILVUS_VECTOR_SIZE", "1024")),
            metric_type=os.getenv("MILVUS_METRIC_TYPE", "COSINE"),
            timeout=int(os.getenv("MILVUS_TIMEOUT", "30")),
            target_chunk_tokens=int(os.getenv("RAG_TARGET_CHUNK_TOKENS", "500")),
            max_chunk_tokens=int(os.getenv("RAG_MAX_CHUNK_TOKENS", "800")),
            overlap_tokens=int(os.getenv("RAG_OVERLAP_TOKENS", "80")),
            enable_rag_vector_outbox=os.getenv("ENABLE_RAG_VECTOR_OUTBOX", "true").lower()
            in {"1", "true", "yes"},
            rag_vector_outbox_max_attempts=int(os.getenv("RAG_VECTOR_OUTBOX_MAX_ATTEMPTS", "5")),
            rag_vector_outbox_processing_timeout_seconds=int(
                os.getenv("RAG_VECTOR_OUTBOX_PROCESSING_TIMEOUT_SECONDS", "900")
            ),
        )
