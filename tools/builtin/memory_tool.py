"""记忆工具：统一入口，按 action 分发到具体记忆操作。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from memory.config import MemoryConfig
from memory.manager import MemoryManager
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
        auto_classify: bool = False,
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
            },
            "required": ["action"],
        }

    def run(self, **kwargs: Any) -> str:
        action = str(kwargs.pop("action", ""))
        return self.execute(action, **kwargs)

    def execute(self, action: str, **kwargs: Any) -> str:
        """执行记忆操作。

        支持的操作：
        - add: 添加记忆（支持4种类型: working/episodic/semantic/perceptual）
        - search: 搜索记忆
        - summary: 获取记忆摘要
        - stats: 获取统计信息
        - update: 更新记忆
        - remove: 删除记忆
        - forget: 遗忘记忆（多种策略）
        - consolidate: 整合记忆（短期→长期）
        - clear_all: 清空所有记忆
        """
        if action == "add":
            return self._add_memory(**kwargs)
        elif action == "search":
            return self._search_memory(**kwargs)
        elif action == "summary":
            return self._get_summary(**kwargs)
        elif action == "stats":
            return self._get_stats(**kwargs)
        elif action == "update":
            return self._update_memory(**kwargs)
        elif action == "remove":
            return self._remove_memory(**kwargs)
        elif action == "forget":
            return self._forget_memory(**kwargs)
        elif action == "consolidate":
            return self._consolidate_memory(**kwargs)
        elif action == "clear_all":
            return self._clear_all_memories(**kwargs)

        return (
            f"错误：不支持的记忆操作 '{action}'。"
            f"支持的操作：{', '.join(SUPPORTED_MEMORY_ACTIONS)}"
        )

    def _add_memory(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        file_path: str | None = None,
        modality: str | None = None,
        **metadata: Any,
    ) -> str:
        """添加记忆。"""
        try:
            if self.memory_manager is None:
                raise RuntimeError("未配置 memory_manager")

            if self.current_session_id is None:
                self.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            if memory_type == "perceptual" and file_path:
                inferred = modality or self._infer_modality(file_path)
                metadata.setdefault("modality", inferred)
                metadata.setdefault("raw_data", file_path)

            metadata.update({
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat(),
            })

            memory_id = self.memory_manager.add_memory(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata,
                auto_classify=False,
            )

            return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"
        except Exception as e:
            return f"❌ 添加记忆失败: {e}"

    def _search_memory(
        self,
        query: str = "",
        memory_type: str = "episodic",
        limit: int = 5,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            if self.memory_manager is None:
                raise RuntimeError("未配置 memory_manager")
            if not query.strip():
                return "❌ 搜索失败: 查询内容不能为空"

            results = self.memory_manager.search_memory(
                query=query,
                memory_type=memory_type,
                limit=limit,
                session_id=self.current_session_id,
            )
            if not results:
                return f"未找到与「{query}」相关的 {memory_type} 记忆"

            lines = [f"找到 {len(results)} 条相关记忆："]
            for index, record in enumerate(results, start=1):
                content = getattr(record, "content", str(record))
                memory_id = getattr(record, "id", "")
                lines.append(f"{index}. [{memory_id[:8]}...] {content}")
            return "\n".join(lines)
        except Exception as e:
            return f"❌ 搜索失败: {e}"

    def _get_summary(self, **kwargs: Any) -> str:
        return self._not_implemented("summary")

    def _get_stats(self, **kwargs: Any) -> str:
        return self._not_implemented("stats")

    def _update_memory(self, **kwargs: Any) -> str:
        return self._not_implemented("update")

    def _remove_memory(self, **kwargs: Any) -> str:
        return self._not_implemented("remove")

    def _forget_memory(self, **kwargs: Any) -> str:
        return self._not_implemented("forget")

    def _consolidate_memory(self, **kwargs: Any) -> str:
        return self._not_implemented("consolidate")

    def _clear_all_memories(self, **kwargs: Any) -> str:
        return self._not_implemented("clear_all")

    def _not_implemented(self, action: str) -> str:
        return f"记忆操作 '{action}' 尚未实现"

    def _validate_memory_types(self, memory_types: list[str]) -> None:
        unsupported = sorted(set(memory_types) - set(SUPPORTED_MEMORY_TYPES))
        if unsupported:
            raise ValueError(f"不支持的记忆类型: {', '.join(unsupported)}")

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
