"""Orphan vector detection tests (C-06)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from memory.orphan_detection import detect_orphan_vectors


def test_no_orphans_when_all_found() -> None:
    """All hit IDs present in found IDs -> no orphans."""
    orphans = detect_orphan_vectors(
        ["mem-1", "mem-2", "mem-3"],
        ["mem-1", "mem-2", "mem-3"],
        memory_kind="episodic",
    )
    assert orphans == []


def test_detects_orphan_vectors() -> None:
    """Hits missing from found IDs are orphans."""
    orphans = detect_orphan_vectors(
        ["mem-1", "mem-2", "mem-3", "mem-4"],
        ["mem-1", "mem-3"],
        memory_kind="episodic",
    )
    assert set(orphans) == {"mem-2", "mem-4"}


def test_orphan_logging_at_warning(caplog: Any) -> None:
    """Orphan detection logs a warning with the orphan IDs."""
    with caplog.at_level(logging.WARNING, logger="memory.orphan_detection"):
        detect_orphan_vectors(
            ["orphan-1", "found-1"],
            ["found-1"],
            memory_kind="semantic",
        )
    assert any("orphan" in record.message.lower() for record in caplog.records)
    assert any("semantic" in record.message for record in caplog.records)


def test_empty_hits_no_orphans() -> None:
    """No hits -> no orphans, no warning."""
    orphans = detect_orphan_vectors([], [], memory_kind="episodic")
    assert orphans == []


def test_truncates_long_orphan_list_in_log(caplog: Any) -> None:
    """More than 10 orphans are truncated in the log message."""
    hit_ids = [f"mem-{i}" for i in range(20)]
    with caplog.at_level(logging.WARNING, logger="memory.orphan_detection"):
        detect_orphan_vectors(hit_ids, [], memory_kind="perceptual")
    warning_messages = [r.message for r in caplog.records if "orphan" in r.message.lower()]
    assert warning_messages
    # Should mention count of 20
    assert "20" in warning_messages[0]


# --- Integration: episodic retrieve filters orphans ---

def test_episodic_retrieve_skips_orphan_vectors() -> None:
    """Episodic retrieve should not return records for orphan vectors."""
    from memory.modules.base import MemoryRecord
    from memory.modules.episodic import EpisodicMemory

    class FakeHit:
        def __init__(self, memory_id: str, score: float) -> None:
            self.memory_id = memory_id
            self.score = score

    class FakeEvent:
        def __init__(self, eid: str) -> None:
            self.id = eid
            self.content = f"content for {eid}"
            self.importance = 0.5
            self.metadata: dict[str, Any] = {}
            self.session_id = None
            self.user_id = "u1"
            self.occurred_at = datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)
            self.created_at = datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc)
            self.sequence_no = 1

    class FakeVectorStore:
        collection_name = "test_collection"
        def search(self, **kwargs: Any) -> list[FakeHit]:
            return [FakeHit("mem-1", 0.9), FakeHit("orphan-1", 0.8)]

    class FakeStore:
        def get_many(self, ids: list[str]) -> list[FakeEvent]:
            # Only return mem-1, orphan-1 is missing
            return [FakeEvent("mem-1")]

    class FakeEmbedding:
        def embed(self, text: str) -> list[float]:
            return [0.1, 0.2]

    from memory.config import MemoryConfig
    config = MemoryConfig()
    memory = EpisodicMemory(
        config=config,
        user_id="u1",
        episodic_store=FakeStore(),
        vector_store=FakeVectorStore(),
        embedding_provider=FakeEmbedding(),
    )

    results = memory.retrieve("query", limit=5)

    assert len(results) == 1
    assert results[0].id == "mem-1"


# --- Integration: perceptual retrieve filters orphans ---

def test_perceptual_retrieve_skips_orphan_vectors() -> None:
    """Perceptual retrieve should not return records for orphan vectors."""
    from memory.modules.perceptual import PerceptualMemory

    class FakeHit:
        def __init__(self, memory_id: str, score: float) -> None:
            self.memory_id = memory_id
            self.score = score

    class FakeItem:
        def __init__(self, item_id: str) -> None:
            self.id = item_id
            self.modality = "text"
            self.content = f"content {item_id}"
            self.importance = 0.5
            self.created_at = datetime.fromtimestamp(1_700_000_000.0, tz=timezone.utc).isoformat()
            self.raw_data = None
            self.metadata: dict[str, Any] = {}
            self.user_id = "u1"

    class FakeVectorStore:
        collection_name = "test_collection"
        def search(self, **kwargs: Any) -> list[FakeHit]:
            return [FakeHit("item-1", 0.9), FakeHit("orphan-x", 0.8)]

    class FakeStore:
        def get_many(self, ids: list[str]) -> list[FakeItem]:
            return [FakeItem("item-1")]
        def get(self, item_id: str) -> FakeItem | None:
            return None

    class FakeEmbedding:
        def embed(self, text: str) -> list[float]:
            return [0.1, 0.2]

    from memory.config import MemoryConfig
    config = MemoryConfig()
    memory = PerceptualMemory(
        config=config,
        user_id="u1",
        perceptual_store=FakeStore(),
        vector_stores={"text": FakeVectorStore()},
        embedding_provider=FakeEmbedding(),
    )

    results = memory.retrieve("query", limit=5)

    assert len(results) == 1
    assert results[0].id == "item-1"
