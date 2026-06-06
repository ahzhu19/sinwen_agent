"""Outbox status aggregation for memory vector indexing."""

from __future__ import annotations

from collections.abc import Mapping

_STATUS_KEYS = ("pending", "processing", "dead")


def collect_outbox_status(
    *,
    pg_outbox: object | None = None,
    semantic_store: object | None = None,
) -> dict[str, dict[str, int]]:
    """Collect pending/processing/dead counts across memory outbox backends."""
    status: dict[str, dict[str, int]] = {}

    if pg_outbox is not None and hasattr(pg_outbox, "status_counts"):
        counts = pg_outbox.status_counts()
        if isinstance(counts, Mapping):
            for kind, values in counts.items():
                status[str(kind)] = _normalize_counts(values)

    if semantic_store is not None and hasattr(semantic_store, "outbox_status_counts"):
        counts = semantic_store.outbox_status_counts()
        if isinstance(counts, Mapping):
            status["semantic"] = _normalize_counts(counts)

    return status


def format_outbox_status(status: dict[str, dict[str, int]]) -> str:
    """Render status counts for CLI output."""
    if not status:
        return "Outbox status: no configured backends"

    lines = ["Outbox status:"]
    for kind in sorted(status):
        counts = _normalize_counts(status[kind])
        lines.append(
            f"  {kind}: pending={counts['pending']} "
            f"processing={counts['processing']} dead={counts['dead']}"
        )
    return "\n".join(lines)


def _normalize_counts(values: Mapping[str, object]) -> dict[str, int]:
    return {key: int(values.get(key) or 0) for key in _STATUS_KEYS}
