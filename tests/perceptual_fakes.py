"""感知记忆测试用 fake 后端。"""

from __future__ import annotations

from dataclasses import dataclass

from tests.episodic_fakes import FakeEmbeddingProvider, FakeVectorStore


@dataclass
class PerceptualFakeBundle:
    text_vectors: FakeVectorStore
    image_vectors: FakeVectorStore
    audio_vectors: FakeVectorStore
    embeddings: FakeEmbeddingProvider

    @property
    def vector_stores(self) -> dict[str, FakeVectorStore]:
        return {
            "text": self.text_vectors,
            "image": self.image_vectors,
            "audio": self.audio_vectors,
        }


def create_perceptual_bundle(vector_size: int = 8) -> PerceptualFakeBundle:
    text_vectors = FakeVectorStore()
    image_vectors = FakeVectorStore()
    audio_vectors = FakeVectorStore()
    text_vectors.collection_name = "fake_perceptual_text_vectors"
    image_vectors.collection_name = "fake_perceptual_image_vectors"
    audio_vectors.collection_name = "fake_perceptual_audio_vectors"
    return PerceptualFakeBundle(
        text_vectors=text_vectors,
        image_vectors=image_vectors,
        audio_vectors=audio_vectors,
        embeddings=FakeEmbeddingProvider(vector_size=vector_size),
    )
