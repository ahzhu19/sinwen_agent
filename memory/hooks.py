"""Agent runtime memory hooks configuration and formatting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_HOOK_SEARCH_MEMORY_TYPES: tuple[str, ...] = ("working", "episodic")


@dataclass
class MemoryHookConfig:
    """Controls automatic memory retrieve/record around Agent runs."""

    enabled: bool = True
    retrieve_before_run: bool = True
    record_after_run: bool = True
    session_id: str | None = None
    search_memory_types: list[str] = field(
        default_factory=lambda: list(DEFAULT_HOOK_SEARCH_MEMORY_TYPES),
    )
    retrieve_limit: int = 3
    record_memory_type: str = "working"
    record_importance: float = 0.5


_MEMORY_TYPE_LABELS = {
    "working": "工作记忆",
    "episodic": "情景记忆",
    "semantic": "语义记忆",
    "perceptual": "感知记忆",
}


def _record_content(record: Any) -> str:
    if hasattr(record, "content"):
        return str(record.content)
    if isinstance(record, dict):
        return str(record.get("content", record))
    return str(record)


def format_retrieved_context(results_by_type: dict[str, list[Any]]) -> str:
    """Format search hits into a system-prompt block."""
    if not results_by_type:
        return ""

    sections: list[str] = ["## 相关记忆（自动检索）"]
    for memory_type, records in results_by_type.items():
        if not records:
            continue
        label = _MEMORY_TYPE_LABELS.get(memory_type, memory_type)
        lines = [f"- {_record_content(record)}" for record in records]
        sections.append(f"### {label}")
        sections.extend(lines)

    if len(sections) == 1:
        return ""
    return "\n".join(sections)


def build_interaction_content(user_message: str, assistant_message: str) -> str:
    """Single working-memory turn: one record keeps Q/A together for retrieval."""
    return f"用户: {user_message.strip()}\n助手: {assistant_message.strip()}"
