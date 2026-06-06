"""记忆系统配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class MemoryConfig:
    """控制记忆模块的基础参数与外部存储连接。"""

    working_memory_capacity: int = 50
    working_memory_ttl_seconds: int = 3600
    episodic_memory_recency_seconds: int = 30 * 24 * 3600
    default_importance: float = 0.5

    database_url: str | None = None
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "hello_agents_episodic_vectors"
    milvus_semantic_collection: str = "hello_agents_semantic_vectors"
    milvus_perceptual_text_collection: str = "hello_agents_perceptual_text_vectors"
    milvus_perceptual_image_collection: str = "hello_agents_perceptual_image_vectors"
    milvus_perceptual_audio_collection: str = "hello_agents_perceptual_audio_vectors"
    milvus_perceptual_video_collection: str = "hello_agents_perceptual_video_vectors"
    milvus_perceptual_file_collection: str = "hello_agents_perceptual_file_vectors"
    milvus_vector_size: int = 1024
    milvus_metric_type: str = "COSINE"
    milvus_timeout: int = 30

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"

    embed_model_type: str = "dashscope"
    embed_model_name: str = "text-embedding-v3"
    embed_api_key: str | None = None
    embed_base_url: str | None = None

    enable_vector_outbox: bool = True
    enable_persistent_vector_outbox: bool = True
    vector_outbox_max_attempts: int = 5
    vector_outbox_poll_on_read: bool = True
    vector_outbox_worker_batch_size: int = 20

    concept_extraction_max_concepts: int = 8
    llm_model_id: str = "gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None

    semantic_graph_max_hops: int = 2
    semantic_graph_hop_decay: float = 0.65
    semantic_graph_expansion_limit: int = 20

    semantic_read_your_writes_limit: int = 20

    @classmethod
    def from_env(cls) -> MemoryConfig:
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
        milvus_uri = os.getenv("MILVUS_URI", f"http://{milvus_host}:{milvus_port}")

        return cls(
            working_memory_capacity=int(os.getenv("WORKING_MEMORY_CAPACITY", "50")),
            working_memory_ttl_seconds=int(os.getenv("WORKING_MEMORY_TTL_SECONDS", "3600")),
            episodic_memory_recency_seconds=int(
                os.getenv("EPISODIC_MEMORY_RECENCY_SECONDS", str(30 * 24 * 3600))
            ),
            default_importance=float(os.getenv("DEFAULT_IMPORTANCE", "0.5")),
            database_url=database_url,
            milvus_uri=milvus_uri,
            milvus_collection=os.getenv(
                "MILVUS_EPISODIC_COLLECTION",
                os.getenv("MILVUS_COLLECTION", "hello_agents_episodic_vectors"),
            ),
            milvus_semantic_collection=os.getenv(
                "MILVUS_SEMANTIC_COLLECTION",
                "hello_agents_semantic_vectors",
            ),
            milvus_perceptual_text_collection=os.getenv(
                "MILVUS_PERCEPTUAL_TEXT_COLLECTION",
                "hello_agents_perceptual_text_vectors",
            ),
            milvus_perceptual_image_collection=os.getenv(
                "MILVUS_PERCEPTUAL_IMAGE_COLLECTION",
                "hello_agents_perceptual_image_vectors",
            ),
            milvus_perceptual_audio_collection=os.getenv(
                "MILVUS_PERCEPTUAL_AUDIO_COLLECTION",
                "hello_agents_perceptual_audio_vectors",
            ),
            milvus_perceptual_video_collection=os.getenv(
                "MILVUS_PERCEPTUAL_VIDEO_COLLECTION",
                "hello_agents_perceptual_video_vectors",
            ),
            milvus_perceptual_file_collection=os.getenv(
                "MILVUS_PERCEPTUAL_FILE_COLLECTION",
                "hello_agents_perceptual_file_vectors",
            ),
            milvus_vector_size=int(os.getenv("MILVUS_VECTOR_SIZE", "1024")),
            milvus_metric_type=os.getenv("MILVUS_METRIC_TYPE", "COSINE"),
            milvus_timeout=int(os.getenv("MILVUS_TIMEOUT", "30")),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            embed_model_type=os.getenv("EMBED_MODEL_TYPE", "dashscope"),
            embed_model_name=os.getenv("EMBED_MODEL_NAME", "text-embedding-v3").strip('"'),
            embed_api_key=os.getenv("EMBED_API_KEY"),
            embed_base_url=os.getenv("EMBED_BASE_URL"),
            enable_vector_outbox=os.getenv("ENABLE_VECTOR_OUTBOX", "true").lower()
            in {"1", "true", "yes"},
            enable_persistent_vector_outbox=os.getenv(
                "ENABLE_PERSISTENT_VECTOR_OUTBOX",
                "true",
            ).lower()
            in {"1", "true", "yes"},
            vector_outbox_max_attempts=int(os.getenv("VECTOR_OUTBOX_MAX_ATTEMPTS", "5")),
            vector_outbox_poll_on_read=os.getenv(
                "VECTOR_OUTBOX_POLL_ON_READ",
                "true",
            ).lower()
            in {"1", "true", "yes"},
            vector_outbox_worker_batch_size=int(
                os.getenv("VECTOR_OUTBOX_WORKER_BATCH_SIZE", "20")
            ),
            concept_extraction_max_concepts=int(
                os.getenv("CONCEPT_EXTRACTION_MAX_CONCEPTS", "8")
            ),
            llm_model_id=os.getenv("LLM_MODEL_ID", "gpt-4o-mini").strip('"'),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL") or os.getenv("EMBED_BASE_URL"),
            semantic_graph_max_hops=int(os.getenv("SEMANTIC_GRAPH_MAX_HOPS", "2")),
            semantic_graph_hop_decay=float(os.getenv("SEMANTIC_GRAPH_HOP_DECAY", "0.65")),
            semantic_graph_expansion_limit=int(
                os.getenv("SEMANTIC_GRAPH_EXPANSION_LIMIT", "20")
            ),
            semantic_read_your_writes_limit=int(
                os.getenv("SEMANTIC_READ_YOUR_WRITES_LIMIT", "20")
            ),
        )
