"""Memory service: internal boundary over MemoryManager.

MemoryTool is only one adapter for this service. Agent runtime hooks should use
this class directly instead of simulating tool calls.
"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig
from .hooks import build_interaction_content, format_retrieved_context
from .manager import MemoryManager
from .protocols import MemoryManagerProtocol


class MemoryService:
    """Application-facing memory API backed by the existing MemoryManager."""

    def __init__(
        self,
        *,
        user_id: str = "default_user",
        config: MemoryConfig | None = None,
        memory_types: list[str] | None = None,
        manager: MemoryManagerProtocol | None = None,
    ) -> None:
        self.user_id = user_id
        self.config = config or MemoryConfig.from_env()
        self.memory_types = list(memory_types or ["working"])
        self._manager = manager or MemoryManager(
            config=self.config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types,
        )

    @property
    def manager(self) -> MemoryManagerProtocol:
        return self._manager

    def add(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        return self._manager.add_memory(content, memory_type, importance, metadata)

    def search(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[Any]:
        return self._manager.search_memory(query, memory_type, limit, session_id)

    def update(
        self,
        memory_id: str,
        memory_type: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._manager.update_memory(
            memory_id,
            memory_type,
            content=content,
            importance=importance,
            metadata=metadata,
        )

    def remove(self, memory_id: str, memory_type: str) -> None:
        self._manager.remove_memory(memory_id, memory_type)

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
        return self._manager.forget_memories(
            memory_type,
            strategy=strategy,
            session_id=session_id,
            importance_threshold=importance_threshold,
            older_than_days=older_than_days,
            limit=limit,
        )

    def consolidate(
        self,
        session_id: str,
        *,
        importance_threshold: float = 0.5,
    ) -> list[str]:
        return self._manager.consolidate_working_to_episodic(
            session_id,
            importance_threshold=importance_threshold,
        )

    def clear(
        self,
        memory_type: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, int]:
        return self._manager.clear_memories(memory_type=memory_type, session_id=session_id)

    def stats(self, session_id: str | None = None) -> dict[str, Any]:
        return self._manager.memory_stats(session_id=session_id)

    def summary(
        self,
        session_id: str | None = None,
        limit_per_type: int = 3,
    ) -> dict[str, list[Any]]:
        return self._manager.memory_summary(
            session_id=session_id,
            limit_per_type=limit_per_type,
        )

    def retrieve_context(
        self,
        query: str,
        *,
        session_id: str | None = None,
        memory_types: list[str] | None = None,
        limit_per_type: int = 3,
    ) -> str:
        requested = memory_types or self.memory_types
        enabled = set(self.memory_types)
        types = [memory_type for memory_type in requested if memory_type in enabled]
        results_by_type: dict[str, list[Any]] = {}
        for memory_type in types:
            results = self.search(
                query,
                memory_type,
                limit=limit_per_type,
                session_id=session_id,
            )
            if results:
                results_by_type[memory_type] = results
        return format_retrieved_context(results_by_type)

    def record_interaction(
        self,
        user_message: str,
        assistant_message: str,
        *,
        session_id: str | None = None,
        memory_type: str = "working",
        importance: float = 0.5,
    ) -> str:
        """Persist one dialog turn as a single working-memory record."""
        metadata: dict[str, Any] = {
            "session_id": session_id or "",
            "source": "agent_hook",
        }
        return self.add(
            build_interaction_content(user_message, assistant_message),
            memory_type,
            importance,
            metadata,
        )
