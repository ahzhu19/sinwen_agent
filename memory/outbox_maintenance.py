"""Outbox 维护：超时回收、死信重放、语义对账补事件。"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig


def run_memory_outbox_maintenance(
    config: MemoryConfig,
    *,
    pg_outbox: Any | None = None,
    episodic_store: Any | None = None,
    semantic_store: Any | None = None,
    reclaim_stale: bool = True,
    replay_dead: bool = False,
    reconcile_semantic: bool = True,
    replay_dead_batch_size: int = 20,
) -> dict[str, int]:
    """运行一轮 outbox 维护，返回各操作影响条数。"""
    results: dict[str, int] = {}
    timeout = config.vector_outbox_processing_timeout_seconds

    if reclaim_stale and pg_outbox is not None and hasattr(pg_outbox, "reclaim_stale_processing"):
        results["pg_reclaimed"] = pg_outbox.reclaim_stale_processing(timeout_seconds=timeout)

    if replay_dead and pg_outbox is not None and hasattr(pg_outbox, "replay_dead"):
        results["pg_replayed_dead"] = pg_outbox.replay_dead(batch_size=replay_dead_batch_size)

    if reclaim_stale and semantic_store is not None and hasattr(
        semantic_store, "reclaim_stale_processing_outbox"
    ):
        results["semantic_reclaimed"] = semantic_store.reclaim_stale_processing_outbox(
            timeout_seconds=timeout,
        )

    if replay_dead and semantic_store is not None and hasattr(semantic_store, "replay_dead_outbox"):
        results["semantic_replayed_dead"] = semantic_store.replay_dead_outbox(
            batch_size=replay_dead_batch_size,
        )

    if reconcile_semantic and semantic_store is not None and hasattr(
        semantic_store, "ensure_pending_outbox_events"
    ):
        results["semantic_outbox_ensured"] = semantic_store.ensure_pending_outbox_events(
            batch_size=config.vector_outbox_worker_batch_size,
            max_attempts=config.vector_outbox_max_attempts,
            collection_name=config.semantic_milvus_collection(),
        )

    if episodic_store is not None and hasattr(episodic_store, "count_unindexed_vectors"):
        results["episodic_unindexed"] = episodic_store.count_unindexed_vectors()

    return results
