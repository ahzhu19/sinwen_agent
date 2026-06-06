"""记忆工具：统一入口，按 action 分发到具体记忆操作。"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.modules.base import MemoryRecord
from tools.base import Tool


SUPPORTED_MEMORY_TYPES = ("working", "episodic", "semantic", "perceptual")
DEFAULT_MEMORY_TYPES = ["working"]
SUPPORTED_MEMORY_ACTIONS = (
    "add",
    "search",
    "summary",
    "stats",
    "update",
    "remove",
    "forget",
    "consolidate",
    "clear_all",
)


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
        session_id: str | None = None,
        importance_threshold: float = 0.2,
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


class MemoryTool(Tool):
    """记忆系统统一工具入口。"""

    def __init__(
        self,
        user_id: str = "default_user",
        session_id: str | None = None,
        memory_config: MemoryConfig | None = None,
        memory_types: list[str] | None = None,
        memory_manager: MemoryManagerProtocol | None = None,
    ) -> None:
        self.user_id = user_id
        self.current_session_id = session_id
        self.memory_config = memory_config or MemoryConfig.from_env()
        self.memory_types = list(memory_types or DEFAULT_MEMORY_TYPES)
        self._validate_memory_types(self.memory_types)
        self.memory_manager = memory_manager or MemoryManager(
            config=self.memory_config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types,
        )

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "管理智能体记忆，支持添加、搜索、摘要、统计、更新、删除、遗忘和整合记忆"

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(SUPPORTED_MEMORY_ACTIONS),
                    "description": "要执行的记忆操作",
                },
                "content": {
                    "type": "string",
                    "description": "要添加或更新的记忆内容",
                },
                "query": {
                    "type": "string",
                    "description": "搜索记忆时使用的查询文本",
                },
                "memory_type": {
                    "type": "string",
                    "enum": list(SUPPORTED_MEMORY_TYPES),
                    "description": "记忆类型",
                },
                "memory_id": {
                    "type": "string",
                    "description": "更新或删除时指定的记忆 ID",
                },
                "importance": {
                    "type": "number",
                    "description": "记忆重要性，范围 0 到 1",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制",
                },
                "file_path": {
                    "type": "string",
                    "description": "感知记忆对应的本地文件路径",
                },
                "modality": {
                    "type": "string",
                    "enum": ["image", "audio", "video", "text", "file", "unknown"],
                    "description": "感知记忆的模态类型",
                },
                "importance_threshold": {
                    "type": "number",
                    "description": "forget/consolidate 使用的重要性阈值",
                },
            },
            "required": ["action"],
        }

    def run(self, **kwargs: Any) -> str:
        action = str(kwargs.pop("action", ""))
        return self.execute(action, **kwargs)

    def execute(self, action: str, **kwargs: Any) -> str:
        if action == "add":
            return self._add_memory(**kwargs)
        if action == "search":
            return self._search_memory(**kwargs)
        if action == "summary":
            return self._get_summary(**kwargs)
        if action == "stats":
            return self._get_stats(**kwargs)
        if action == "update":
            return self._update_memory(**kwargs)
        if action == "remove":
            return self._remove_memory(**kwargs)
        if action == "forget":
            return self._forget_memory(**kwargs)
        if action == "consolidate":
            return self._consolidate_memory(**kwargs)
        if action == "clear_all":
            return self._clear_all_memories(**kwargs)
        return (
            f"错误：不支持的记忆操作 '{action}'。"
            f"支持的操作：{', '.join(SUPPORTED_MEMORY_ACTIONS)}"
        )

    def _ensure_session_id(self) -> str:
        if self.current_session_id is None:
            self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return self.current_session_id

    def _add_memory(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        file_path: str | None = None,
        modality: str | None = None,
        **metadata: Any,
    ) -> str:
        try:
            session_id = self._ensure_session_id()
            payload = dict(metadata)

            if memory_type == "perceptual" and file_path:
                inferred = modality or self._infer_modality(file_path)
                payload.setdefault("modality", inferred)
                payload.setdefault("raw_data", file_path)

            payload.setdefault("session_id", session_id)
            payload.setdefault("timestamp", datetime.now().isoformat())

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=payload,
            )
            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"
        except Exception as exc:
            return f"❌ 添加记忆失败: {exc}"

    def _search_memory(
        self,
        query: str = "",
        memory_type: str = "working",
        limit: int = 5,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            if not query.strip():
                return "❌ 搜索失败: 查询内容不能为空"
            if memory_type not in self.memory_types:
                return f"❌ 搜索失败: 未启用记忆类型 {memory_type}"

            results = self.memory_manager.search_memory(
                query=query,
                memory_type=memory_type,
                limit=limit,
                session_id=self.current_session_id,
            )
            if not results:
                return f"未找到与「{query}」相关的 {memory_type} 记忆"
            return self._format_records(f"找到 {len(results)} 条相关记忆", results)
        except Exception as exc:
            return f"❌ 搜索失败: {exc}"

    def _get_summary(self, limit: int = 3, **kwargs: Any) -> str:
        _ = kwargs
        try:
            session_id = self._ensure_session_id()
            summary = self.memory_manager.memory_summary(
                session_id=session_id,
                limit_per_type=limit,
            )
            if not summary:
                return "当前会话暂无可用记忆摘要"
            lines = [f"会话 {session_id} 记忆摘要："]
            for memory_type, records in summary.items():
                lines.append(f"\n[{memory_type}]")
                if not records:
                    lines.append("  (无)")
                    continue
                for index, record in enumerate(records, start=1):
                    lines.append(self._format_record_line(index, record))
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ 摘要失败: {exc}"

    def _get_stats(self, **kwargs: Any) -> str:
        _ = kwargs
        try:
            stats = self.memory_manager.memory_stats(session_id=self.current_session_id)
            counts = stats.get("counts", {})
            parts = [
                f"用户 {stats.get('user_id')} 记忆统计",
                f"已启用类型: {', '.join(stats.get('enabled_types', []))}",
            ]
            for memory_type, count in counts.items():
                if count is None:
                    parts.append(f"- {memory_type}: 已启用（暂无计数接口）")
                else:
                    parts.append(f"- {memory_type}: {count} 条")
            return "\n".join(parts)
        except Exception as exc:
            return f"❌ 统计失败: {exc}"

    def _update_memory(
        self,
        memory_id: str = "",
        memory_type: str = "working",
        content: str = "",
        importance: float | None = None,
        **metadata: Any,
    ) -> str:
        try:
            if not memory_id.strip():
                return "❌ 更新失败: memory_id 不能为空"
            updated_id = self.memory_manager.update_memory(
                memory_id.strip(),
                memory_type,
                content=content or None,
                importance=importance,
                metadata=dict(metadata) if metadata else None,
            )
            return f"✅ 记忆已更新 (ID: {updated_id[:8]}...)"
        except Exception as exc:
            return f"❌ 更新失败: {exc}"

    def _remove_memory(
        self,
        memory_id: str = "",
        memory_type: str = "working",
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            if not memory_id.strip():
                return "❌ 删除失败: memory_id 不能为空"
            self.memory_manager.remove_memory(memory_id.strip(), memory_type)
            return f"✅ 已删除 {memory_type} 记忆 {memory_id[:8]}..."
        except Exception as exc:
            return f"❌ 删除失败: {exc}"

    def _forget_memory(
        self,
        memory_type: str = "working",
        importance_threshold: float = 0.2,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            removed = self.memory_manager.forget_memories(
                memory_type,
                session_id=self.current_session_id,
                importance_threshold=importance_threshold,
            )
            return f"✅ 已遗忘 {removed} 条 {memory_type} 记忆（阈值 <= {importance_threshold}）"
        except Exception as exc:
            return f"❌ 遗忘失败: {exc}"

    def _consolidate_memory(
        self,
        importance_threshold: float = 0.5,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            session_id = self._ensure_session_id()
            created_ids = self.memory_manager.consolidate_working_to_episodic(
                session_id,
                importance_threshold=importance_threshold,
            )
            if not created_ids:
                return "未找到需要整合到情景记忆的工作记忆"
            return (
                f"✅ 已将 {len(created_ids)} 条工作记忆整合为情景记忆 "
                f"(importance >= {importance_threshold})"
            )
        except Exception as exc:
            return f"❌ 整合失败: {exc}"

    def _clear_all_memories(
        self,
        memory_type: str | None = None,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            cleared = self.memory_manager.clear_memories(
                memory_type=memory_type or None,
                session_id=self.current_session_id,
            )
            if not cleared:
                return "没有可清空的记忆"
            lines = ["✅ 已清空记忆："]
            for cleared_type, count in cleared.items():
                lines.append(f"- {cleared_type}: {count} 条")
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ 清空失败: {exc}"

    def _format_records(self, title: str, records: list[Any]) -> str:
        lines = [f"{title}："]
        for index, record in enumerate(records, start=1):
            lines.append(self._format_record_line(index, record))
        return "\n".join(lines)

    def _format_record_line(self, index: int, record: Any) -> str:
        if isinstance(record, MemoryRecord):
            memory_id = record.id
            content = record.content
        else:
            memory_id = getattr(record, "id", "")
            content = getattr(record, "content", str(record))
        return f"{index}. [{memory_id[:8]}...] {content}"

    def _validate_memory_types(self, memory_types: list[str]) -> None:
        unsupported = sorted(set(memory_types) - set(SUPPORTED_MEMORY_TYPES))
        if unsupported:
            raise ValueError(f"不支持的记忆类型: {', '.join(unsupported)}")
        if "perceptual" in memory_types:
            warnings.warn(
                "perceptual 记忆仍为 experimental（进程内存元数据 + 文本代理 embedding），"
                "生产环境请优先 semantic 或 RAG",
                stacklevel=3,
            )

    def _infer_modality(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}:
            return "image"
        if suffix in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
            return "audio"
        if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            return "video"
        if suffix in {".txt", ".md", ".pdf", ".doc", ".docx"}:
            return "text"
        return "unknown"
