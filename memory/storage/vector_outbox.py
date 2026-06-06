"""Milvus 向量双写失败时的内存 outbox 与补偿重试。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

VectorOutboxKind = Literal["episodic"]


class VectorWriteError(RuntimeError):
    """Milvus 写入失败且无法入队 outbox。"""


@dataclass
class VectorOutboxEntry:
    kind: VectorOutboxKind
    memory_id: str
    vector: list[float]
    user_id: str
    session_id: str | None
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    last_error: str | None = None


class VectorOutbox:
    """记录结构化/图存储已成功但 Milvus upsert 失败的条目。"""

    def __init__(self, *, max_attempts: int = 5) -> None:
        self._entries: dict[tuple[VectorOutboxKind, str], VectorOutboxEntry] = {}
        self._max_attempts = max_attempts

    def pending_count(self) -> int:
        return len(self._entries)

    def enqueue(
        self,
        kind: VectorOutboxKind,
        *,
        memory_id: str,
        vector: list[float],
        user_id: str,
        session_id: str | None,
        error: str,
    ) -> None:
        key = (kind, memory_id)
        existing = self._entries.get(key)
        attempts = existing.attempts if existing else 0
        self._entries[key] = VectorOutboxEntry(
            kind=kind,
            memory_id=memory_id,
            vector=list(vector),
            user_id=user_id,
            session_id=session_id,
            created_at=existing.created_at if existing else time.time(),
            attempts=attempts,
            last_error=error,
        )

    def flush(self, vector_store: Any, kind: VectorOutboxKind) -> tuple[int, int]:
        """重试指定类型的 pending 条目。返回 (成功数, 失败数)。"""
        succeeded = 0
        failed = 0
        keys = [key for key in self._entries if key[0] == kind]
        for key in keys:
            entry = self._entries[key]
            if entry.attempts >= self._max_attempts:
                failed += 1
                continue
            entry.attempts += 1
            try:
                vector_store.upsert(
                    memory_id=entry.memory_id,
                    vector=entry.vector,
                    user_id=entry.user_id,
                    session_id=entry.session_id,
                )
            except Exception as exc:  # noqa: BLE001 — 保留条目供下次补偿
                entry.last_error = str(exc)
                failed += 1
                continue
            del self._entries[key]
            succeeded += 1
        return succeeded, failed

    def snapshot(self) -> list[VectorOutboxEntry]:
        return list(self._entries.values())


def upsert_vector_with_outbox(
    *,
    outbox: VectorOutbox | None,
    kind: VectorOutboxKind,
    vector_store: Any,
    memory_id: str,
    vector: list[float],
    user_id: str,
    session_id: str | None,
) -> bool:
    """写入 Milvus；失败时入队 outbox 而不抛错（结构化存储已提交）。"""
    try:
        vector_store.upsert(
            memory_id=memory_id,
            vector=vector,
            user_id=user_id,
            session_id=session_id,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        if outbox is not None:
            outbox.enqueue(
                kind,
                memory_id=memory_id,
                vector=vector,
                user_id=user_id,
                session_id=session_id,
                error=str(exc),
            )
            return False
        raise VectorWriteError(
            f"Milvus upsert 失败且未配置 outbox（memory_id={memory_id}, kind={kind}）: {exc}"
        ) from exc
