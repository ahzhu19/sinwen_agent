"""记忆操作 Mixin：CRUD / forget / consolidate / stats / outbox。

从 MemoryManager 抽取，依赖实例属性：
  memory_modules, config, user_id, store,
  vector_outbox, pg_vector_outbox, _outbox_processor
"""

from __future__ import annotations

import copy
from typing import Any

from .forget_policy import (
    default_forget_limit,
    default_forget_threshold,
    parse_occurred_at,
    should_forget_record,
)
from .modules.base import MemoryRecord
from .records import (
    episodic_event_to_record,
    perceptual_item_to_record,
    semantic_fact_to_record,
)

_VECTOR_MEMORY_TYPES = frozenset({"episodic", "semantic", "perceptual"})


class MemoryOperations:
    """CRUD / forget / consolidate / stats / outbox 操作。

    设计为 mixin：MemoryManager 继承此类，__init__ 负责设置实例属性。
    """

    # 实例属性声明（由 MemoryManager.__init__ 设置）
    memory_modules: dict[str, Any]
    config: Any
    user_id: str
    store: Any
    vector_outbox: Any
    pg_vector_outbox: Any
    _outbox_processor: Any

    # -- CRUD ----------------------------------------------------------

    def add_memory(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")

        return memory_module.add(
            content=content,
            importance=importance,
            metadata=metadata,
        )

    def search_memory(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[Any]:
        if memory_type in _VECTOR_MEMORY_TYPES and self.config.vector_outbox_poll_on_read:
            self.flush_vector_outbox()
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")
        if not hasattr(memory_module, "retrieve"):
            raise ValueError(f"记忆类型 '{memory_type}' 不支持检索")
        return memory_module.retrieve(query=query, limit=limit, session_id=session_id)

    def remove_memory(self, memory_id: str, memory_type: str) -> None:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")
        if not hasattr(memory_module, "remove"):
            raise ValueError(f"记忆类型 '{memory_type}' 不支持删除")
        memory_module.remove(memory_id)

    def update_memory(
        self,
        memory_id: str,
        memory_type: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")

        if memory_type == "working":
            if not hasattr(memory_module, "update"):
                raise ValueError(f"记忆类型 '{memory_type}' 不支持更新")
            updated = memory_module.update(
                memory_id,
                content=content,
                importance=importance,
                metadata=metadata,
            )
            if updated is None:
                raise KeyError(f"未找到记忆: {memory_id}")
            return updated.id

        existing = self._get_record(memory_type, memory_id)
        if existing is None:
            raise KeyError(f"未找到记忆: {memory_id}")

        merged_metadata = copy.deepcopy(existing.metadata)
        if metadata:
            merged_metadata.update(metadata)
        new_content = content if content is not None else existing.content
        new_importance = importance if importance is not None else existing.importance

        if not hasattr(memory_module, "update"):
            raise ValueError(f"记忆类型 '{memory_type}' 不支持更新")
        return memory_module.update(
            memory_id,
            content=new_content,
            importance=new_importance,
            metadata=merged_metadata,
        )

    # -- Stats / Summary ----------------------------------------------

    def memory_stats(self, session_id: str | None = None) -> dict[str, Any]:
        stats: dict[str, Any] = {"user_id": self.user_id, "enabled_types": list(self.memory_modules)}
        counts: dict[str, int] = {}

        if "working" in self.memory_modules:
            records = self.memory_modules["working"].store.list_records(memory_type="working")
            if session_id:
                records = [r for r in records if r.metadata.get("session_id") == session_id]
            counts["working"] = len(records)

        if "episodic" in self.memory_modules:
            timeline = self.memory_modules["episodic"].list_timeline(
                session_id=session_id,
                limit=10_000,
            )
            counts["episodic"] = len(timeline)

        if "perceptual" in self.memory_modules:
            store = self.memory_modules["perceptual"]._store
            items = [
                item
                for item in getattr(store, "items", {}).values()
                if getattr(item, "user_id", None) == self.user_id
            ]
            if session_id:
                items = [item for item in items if item.metadata.get("session_id") == session_id]
            counts["perceptual"] = len(items)

        if "semantic" in self.memory_modules:
            counts["semantic"] = self.memory_modules["semantic"].count_for_user(session_id)

        stats["counts"] = counts
        return stats

    def memory_summary(
        self,
        session_id: str | None = None,
        limit_per_type: int = 3,
    ) -> dict[str, list[Any]]:
        summary: dict[str, list[Any]] = {}
        if "working" in self.memory_modules and session_id:
            summary["working"] = self.memory_modules["working"].list_recent(session_id)[
                -limit_per_type:
            ]
        if "episodic" in self.memory_modules:
            summary["episodic"] = self.memory_modules["episodic"].list_timeline(
                session_id=session_id,
                limit=limit_per_type,
            )
        if "semantic" in self.memory_modules:
            summary["semantic"] = self.memory_modules["semantic"].list_for_user(
                session_id=session_id,
                limit=limit_per_type,
            )
        return summary

    # -- Forget --------------------------------------------------------

    def forget_memories(
        self,
        memory_type: str = "working",
        *,
        strategy: str = "importance",
        session_id: str | None = None,
        importance_threshold: float | None = None,
        older_than_days: int | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> int | list[MemoryRecord]:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            raise ValueError(f"未启用记忆类型: {memory_type}")

        threshold = (
            importance_threshold
            if importance_threshold is not None
            else default_forget_threshold(memory_type)
        )
        max_remove = limit if limit is not None else default_forget_limit(memory_type)

        candidates = self._collect_forget_candidates(
            memory_type,
            memory_module,
            strategy=strategy,
            session_id=session_id,
            threshold=threshold,
            older_than_days=older_than_days,
            max_remove=max_remove,
        )

        if dry_run:
            return candidates

        for record in candidates:
            memory_module.remove(record.id)
        return len(candidates)

    def _collect_forget_candidates(
        self,
        memory_type: str,
        memory_module: Any,
        *,
        strategy: str,
        session_id: str | None,
        threshold: float,
        older_than_days: int | None,
        max_remove: int,
    ) -> list[MemoryRecord]:
        """收集符合 forget 条件的候选，统一转为 MemoryRecord，不执行删除。"""
        candidates: list[MemoryRecord] = []

        if memory_type == "working":
            memory_module.cleanup_expired()
            for record in list(memory_module.store.list_records(memory_type="working")):
                if len(candidates) >= max_remove:
                    break
                if session_id and record.metadata.get("session_id") != session_id:
                    continue
                if should_forget_record(
                    importance=record.importance,
                    importance_threshold=threshold,
                    strategy=strategy,
                ):
                    candidates.append(record)
            return candidates

        if memory_type == "episodic":
            store = memory_module._store
            if not hasattr(store, "list_for_forget"):
                raise ValueError("episodic store 需实现 list_for_forget")
            for event in store.list_for_forget(self.user_id, session_id=session_id, limit=10_000):
                if len(candidates) >= max_remove:
                    break
                if should_forget_record(
                    importance=event.importance,
                    importance_threshold=threshold,
                    strategy=strategy,
                    occurred_at=event.occurred_at,
                    older_than_days=older_than_days,
                ):
                    candidates.append(episodic_event_to_record(event))
            return candidates

        if memory_type == "semantic":
            facts = memory_module.list_for_user(session_id=session_id, limit=10_000)
            for record in facts:
                if len(candidates) >= max_remove:
                    break
                if should_forget_record(
                    importance=record.importance,
                    importance_threshold=threshold,
                    strategy=strategy,
                    occurred_at=parse_occurred_at(record),
                    older_than_days=older_than_days,
                ):
                    candidates.append(semantic_fact_to_record(record))
            return candidates

        if memory_type == "perceptual":
            store = memory_module._store
            if not hasattr(store, "list_by_user"):
                raise ValueError("perceptual store 需实现 list_by_user")
            for item in store.list_by_user(self.user_id, session_id=session_id, limit=10_000):
                if len(candidates) >= max_remove:
                    break
                if should_forget_record(
                    importance=item.importance,
                    importance_threshold=threshold,
                    strategy=strategy,
                    occurred_at=parse_occurred_at(item),
                    older_than_days=older_than_days,
                ):
                    candidates.append(perceptual_item_to_record(item))
            return candidates

        raise ValueError(f"记忆类型 '{memory_type}' 暂不支持 forget 策略")

    # -- Consolidate ---------------------------------------------------

    def consolidate_working_to_episodic(
        self,
        session_id: str,
        *,
        importance_threshold: float = 0.5,
    ) -> list[str]:
        if "working" not in self.memory_modules:
            raise ValueError("未启用 working 记忆")
        if "episodic" not in self.memory_modules:
            raise ValueError("未启用 episodic 记忆，无法整合")

        working = self.memory_modules["working"]
        episodic = self.memory_modules["episodic"]
        created_ids: list[str] = []
        for record in working.list_recent(session_id):
            if record.importance < importance_threshold:
                continue
            memory_id = episodic.add(
                content=record.content,
                importance=record.importance,
                metadata={
                    **copy.deepcopy(record.metadata),
                    "consolidated_from": record.id,
                    "session_id": session_id,
                },
            )
            working.remove(record.id)
            created_ids.append(memory_id)
        return created_ids

    # -- Clear ---------------------------------------------------------

    def clear_memories(
        self,
        memory_type: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, int]:
        targets = [memory_type] if memory_type else list(self.memory_modules)
        cleared: dict[str, int] = {}

        for target in targets:
            module = self.memory_modules.get(target)
            if module is None:
                continue

            if target == "working":
                if session_id:
                    before = len(module.store.list_records(memory_type="working"))
                    module.clear_session(session_id)
                    after = len(module.store.list_records(memory_type="working"))
                    cleared[target] = before - after
                else:
                    records = list(module.store.list_records(memory_type="working"))
                    for record in records:
                        module.remove(record.id)
                    cleared[target] = len(records)
                continue

            if target == "episodic":
                timeline = module.list_timeline(session_id=session_id, limit=10_000)
                for record in timeline:
                    module.remove(record.id)
                cleared[target] = len(timeline)
                continue

            if target == "perceptual":
                store = module._store
                items = list(getattr(store, "items", {}).values())
                count = 0
                for item in items:
                    if item.user_id != self.user_id:
                        continue
                    if session_id and item.metadata.get("session_id") != session_id:
                        continue
                    module.remove(item.id)
                    count += 1
                cleared[target] = count
                continue

            if target == "semantic":
                cleared[target] = self.memory_modules["semantic"].remove_all_for_user(
                    session_id
                )
                continue

        return cleared

    # -- Outbox --------------------------------------------------------

    def flush_vector_outbox(self) -> dict[str, tuple[int, int]]:
        """补偿 Milvus 双写失败条目。返回各类型 (成功数, 仍失败数)。"""
        batch_size = self.config.vector_outbox_worker_batch_size
        results: dict[str, tuple[int, int]] = {}

        if self._outbox_processor is not None:
            for kind in ("episodic", "perceptual"):
                if kind in self.memory_modules:
                    results[kind] = self._outbox_processor.process_batch(
                        batch_size=batch_size,
                        memory_kind=kind,
                    )

        if "semantic" in self.memory_modules and hasattr(
            self.memory_modules["semantic"],
            "flush_vector_outbox",
        ):
            results["semantic"] = self.memory_modules["semantic"].flush_vector_outbox()

        if results:
            return results

        if "episodic" in self.memory_modules and hasattr(
            self.memory_modules["episodic"],
            "flush_vector_outbox",
        ):
            results["episodic"] = self.memory_modules["episodic"].flush_vector_outbox()
        return results

    def vector_outbox_pending(self) -> int:
        pending = 0
        if self.pg_vector_outbox is not None:
            pending += self.pg_vector_outbox.pending_count()
        semantic = self.memory_modules.get("semantic")
        if semantic is not None:
            store = getattr(semantic, "_store", None)
            if store is not None and hasattr(store, "pending_outbox_count"):
                pending += store.pending_outbox_count()
        if pending:
            return pending
        if self.vector_outbox is None:
            return 0
        return self.vector_outbox.pending_count()

    # -- Helpers -------------------------------------------------------

    def _get_record(self, memory_type: str, memory_id: str) -> Any | None:
        memory_module = self.memory_modules.get(memory_type)
        if memory_module is None:
            return None
        if hasattr(memory_module, "get"):
            return memory_module.get(memory_id)
        return None
