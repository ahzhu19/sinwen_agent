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
        self.stats_counts = stats_counts or {"working": 2}

    def add_memory(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.memory_id

    def search_memory(self, **kwargs: Any) -> list[Any]:
        _ = kwargs
        return []

    def remove_memory(self, memory_id: str, memory_type: str) -> None:
        _ = memory_id, memory_type

    def update_memory(self, memory_id: str, memory_type: str, **kwargs: Any) -> str:
        _ = memory_type, kwargs
        return memory_id

    def memory_stats(self, session_id: str | None = None) -> dict[str, Any]:
        _ = session_id
        return {
            "user_id": "u1",
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
        return 0

    def consolidate_working_to_episodic(
        self,
        session_id: str,
        **kwargs: Any,
    ) -> list[str]:
        _ = session_id, kwargs
        return []

    def clear_memories(
        self,
        memory_type: str | None = None,
        **kwargs: Any,
    ) -> dict[str, int]:
        _ = memory_type, kwargs
        return {}


class FailingMemoryManager:
    def add_memory(self, **kwargs: Any) -> str:
        _ = kwargs
        raise RuntimeError("database unavailable")
