"""记忆系统 Protocol 定义（Manager / Module 共享接口）。"""

from __future__ import annotations

from typing import Any, Protocol

from .modules.base import MemoryRecord


class MemoryModuleProtocol(Protocol):
    memory_type: str

    def add(
        self,
        content: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        ...

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> list[MemoryRecord]:
        ...

    def remove(self, memory_id: str) -> None:
        ...

    def get(self, memory_id: str) -> MemoryRecord | None:
        ...


class MemoryManagerProtocol(Protocol):
    def add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        ...

    def search_memory(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[Any]:
        ...

    def remove_memory(self, memory_id: str, memory_type: str) -> None:
        ...

    def update_memory(
        self,
        memory_id: str,
        memory_type: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        ...

    def memory_stats(self, session_id: str | None = None) -> dict[str, Any]:
        ...

    def memory_summary(
        self,
        session_id: str | None = None,
        limit_per_type: int = 3,
    ) -> dict[str, list[Any]]:
        ...

    def forget_memories(
        self,
        memory_type: str = "working",
        *,
        strategy: str = "importance",
        session_id: str | None = None,
        importance_threshold: float | None = None,
        older_than_days: int | None = None,
        limit: int | None = None,
    ) -> int:
        ...

    def consolidate_working_to_episodic(
        self,
        session_id: str,
        *,
        importance_threshold: float = 0.5,
    ) -> list[str]:
        ...

    def clear_memories(
        self,
        memory_type: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, int]:
        ...


class MemoryServiceProtocol(Protocol):
    def add(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        ...

    def search(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[Any]:
        ...

    def update(
        self,
        memory_id: str,
        memory_type: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        ...

    def remove(self, memory_id: str, memory_type: str) -> None:
        ...

    def forget(
        self,
        memory_type: str = "working",
        *,
        strategy: str = "importance",
        session_id: str | None = None,
        importance_threshold: float | None = None,
        older_than_days: int | None = None,
        limit: int | None = None,
    ) -> int:
        ...

    def consolidate(
        self,
        session_id: str,
        *,
        importance_threshold: float = 0.5,
    ) -> list[str]:
        ...

    def clear(
        self,
        memory_type: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, int]:
        ...

    def stats(self, session_id: str | None = None) -> dict[str, Any]:
        ...

    def summary(
        self,
        session_id: str | None = None,
        limit_per_type: int = 3,
    ) -> dict[str, list[Any]]:
        ...

    def retrieve_context(
        self,
        query: str,
        *,
        session_id: str | None = None,
        memory_types: list[str] | None = None,
        limit_per_type: int = 3,
    ) -> str:
        ...

    def record_interaction(
        self,
        user_message: str,
        assistant_message: str,
        *,
        session_id: str | None = None,
        memory_type: str = "working",
        importance: float = 0.5,
    ) -> str:
        ...
