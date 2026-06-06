"""记忆 forget 策略。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_FORGET_THRESHOLDS: dict[str, float] = {
    "working": 0.2,
    "episodic": 0.2,
    "semantic": 0.15,
    "perceptual": 0.2,
}

DEFAULT_FORGET_LIMITS: dict[str, int] = {
    "working": 500,
    "episodic": 200,
    "semantic": 50,
    "perceptual": 100,
}


def default_forget_threshold(memory_type: str) -> float:
    return DEFAULT_FORGET_THRESHOLDS.get(memory_type, 0.2)


def default_forget_limit(memory_type: str) -> int:
    return DEFAULT_FORGET_LIMITS.get(memory_type, 100)


def should_forget_record(
    *,
    importance: float,
    importance_threshold: float,
    strategy: str,
    occurred_at: datetime | None = None,
    older_than_days: int | None = None,
) -> bool:
    if importance > importance_threshold:
        return False
    if strategy == "importance_ttl" and older_than_days is not None and occurred_at is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=timezone.utc)
        return occurred_at <= cutoff
    if strategy in {"importance", "session"}:
        return True
    raise ValueError(f"不支持的 forget strategy: {strategy}")


def parse_occurred_at(record: Any) -> datetime | None:
    if hasattr(record, "occurred_at"):
        value = record.occurred_at
        if isinstance(value, datetime):
            return value
    metadata = getattr(record, "metadata", {}) or {}
    for key in ("occurred_at", "created_at"):
        raw = metadata.get(key)
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, (int, float)):
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                continue
    created_at = getattr(record, "created_at", None)
    if isinstance(created_at, str):
        try:
            return datetime.fromisoformat(created_at)
        except ValueError:
            return None
    return None
