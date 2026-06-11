"""Shared memory hook helpers for Agent runtimes."""

from __future__ import annotations

from memory.hooks import MemoryHookConfig
from memory.protocols import MemoryServiceProtocol


def resolve_memory_context(
    memory_service: MemoryServiceProtocol | None,
    memory_hooks: MemoryHookConfig | None,
    query: str,
) -> str | None:
    if memory_service is None or memory_hooks is None or not memory_hooks.enabled:
        return None
    if not memory_hooks.retrieve_before_run:
        return None

    context = memory_service.retrieve_context(
        query,
        session_id=memory_hooks.session_id,
        memory_types=memory_hooks.search_memory_types,
        limit_per_type=memory_hooks.retrieve_limit,
    )
    return context or None


def maybe_record_interaction(
    memory_service: MemoryServiceProtocol | None,
    memory_hooks: MemoryHookConfig | None,
    user_message: str,
    assistant_message: str,
) -> str | None:
    if memory_service is None or memory_hooks is None or not memory_hooks.enabled:
        return None
    if not memory_hooks.record_after_run:
        return None
    if not assistant_message.strip():
        return None

    return memory_service.record_interaction(
        user_message,
        assistant_message,
        session_id=memory_hooks.session_id,
        memory_type=memory_hooks.record_memory_type,
        importance=memory_hooks.record_importance,
    )


def append_memory_context(system_prompt: str | None, memory_context: str | None) -> str | None:
    if not memory_context:
        return system_prompt
    if system_prompt:
        return f"{system_prompt}\n\n{memory_context}"
    return memory_context


def resolve_memory_hooks_enabled(
    enable_memory: bool,
    enable_memory_hooks: bool | None,
) -> bool:
    """Default: hooks follow ``enable_memory`` unless explicitly overridden."""
    if enable_memory_hooks is not None:
        return enable_memory_hooks
    return enable_memory


def build_memory_hook_config(session_id: str | None) -> MemoryHookConfig:
    return MemoryHookConfig(session_id=session_id)
