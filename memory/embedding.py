"""Embedding 提供方：从 EMBED_* 环境变量读取 DashScope/OpenAI-compatible 配置。"""

from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from .config import MemoryConfig


class EmbeddingProvider(Protocol):
    """文本向量化接口。"""

    @property
    def vector_size(self) -> int:
        ...

    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAICompatibleEmbeddingProvider:
    """通过 OpenAI-compatible API 调用 embedding（含 DashScope compatible-mode）。"""

    def __init__(self, config: MemoryConfig) -> None:
        if not config.embed_api_key:
            raise ValueError("未配置 EMBED_API_KEY，无法生成 embedding 向量")
        if not config.embed_base_url:
            raise ValueError("未配置 EMBED_BASE_URL，无法生成 embedding 向量")

        self._model = config.embed_model_name
        self._vector_size = config.milvus_vector_size
        self._client = OpenAI(
            api_key=config.embed_api_key,
            base_url=config.embed_base_url,
            timeout=60,
        )

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        for vector in vectors:
            if len(vector) != self._vector_size:
                raise ValueError(
                    f"embedding 维度 {len(vector)} 与 MILVUS_VECTOR_SIZE={self._vector_size} 不一致"
                )
        return vectors


def create_embedding_provider(config: MemoryConfig) -> EmbeddingProvider:
    model_type = config.embed_model_type.lower()
    if model_type in {"dashscope", "openai", "compatible"}:
        return OpenAICompatibleEmbeddingProvider(config)
    raise ValueError(f"不支持的 EMBED_MODEL_TYPE: {config.embed_model_type}")
