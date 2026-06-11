"""Shared fake memory backends for unit tests."""

from __future__ import annotations

from typing import Any


class FakeMemoryManager:
    def __init__(
        self,
        memory_id: str = "mem-12345678-abcd",
        *,
        stats_counts: dict[str, int] | None = None,
    ) -> None:
        self.memory_id = memory_id
        self.calls: list[dict[str, Any]] = []
        self.added: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []
        self.removed: list[tuple[str, str]] = []
        self.forgotten_count = 0
        self.consolidated_ids: list[str] = []
        self.cleared: dict[str, int] = {}
        self.stats_user_id = "u1"
        self.stats_counts = stats_counts or {"working": 2}

    def add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        payload = {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "metadata": metadata,
            **kwargs,
        }
        self.calls.append(payload)
        self.added.append(payload)
        return self.memory_id

    def search_memory(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        payload = {
            "query": query,
            "memory_type": memory_type,
            "limit": limit,
            "session_id": session_id,
            **kwargs,
        }
        self.searches.append(payload)
        return [{"id": self.memory_id, "content": "fake"}]

    def remove_memory(self, memory_id: str, memory_type: str) -> None:
        self.removed.append((memory_id, memory_type))

    def update_memory(
        self,
        memory_id: str,
        memory_type: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "memory_id": memory_id,
            "memory_type": memory_type,
            "content": content,
            "importance": importance,
            "metadata": metadata,
            **kwargs,
        }
        self.updated.append(payload)
        return memory_id

    def memory_stats(self, session_id: str | None = None) -> dict[str, Any]:
        _ = session_id
        return {
            "user_id": self.stats_user_id,
            "enabled_types": ["working"],
            "counts": dict(self.stats_counts),
        }

    def memory_summary(
        self,
        session_id: str | None = None,
        limit_per_type: int = 3,
    ) -> dict[str, list[Any]]:
        _ = session_id, limit_per_type
        return {"working": []}

    def forget_memories(self, memory_type: str = "working", **kwargs: Any) -> int:
        _ = memory_type, kwargs
        return self.forgotten_count

    def consolidate_working_to_episodic(
        self,
        session_id: str,
        **kwargs: Any,
    ) -> list[str]:
        _ = session_id, kwargs
        return list(self.consolidated_ids)

    def clear_memories(
        self,
        memory_type: str | None = None,
        **kwargs: Any,
    ) -> dict[str, int]:
        _ = memory_type, kwargs
        return dict(self.cleared)


class FailingMemoryManager:
    def add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        _ = content, memory_type, importance, metadata
        raise RuntimeError("database unavailable")
