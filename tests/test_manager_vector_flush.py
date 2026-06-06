"""MemoryManager 向量 outbox 读路径补偿。"""

from __future__ import annotations

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.storage.document_store import InMemoryPerceptualStore
from tests.perceptual_fakes import create_perceptual_bundle


def test_search_memory_flushes_perceptual_outbox_when_poll_on_read(monkeypatch) -> None:
    bundle = create_perceptual_bundle()
    config = MemoryConfig(vector_outbox_poll_on_read=True)
    manager = MemoryManager(
        config=config,
        user_id="user123",
        enable_working=False,
        enable_episodic=False,
        enable_semantic=False,
        enable_perceptual=True,
        perceptual_store=InMemoryPerceptualStore(),
        perceptual_vector_stores=bundle.vector_stores,
        perceptual_embedding_provider=bundle.embeddings,
    )

    flushed: list[str] = []

    def fake_flush() -> dict[str, tuple[int, int]]:
        flushed.append("called")
        return {}

    monkeypatch.setattr(manager, "flush_vector_outbox", fake_flush)

    manager.search_memory(query="白板", memory_type="perceptual", limit=3)

    assert flushed == ["called"]
