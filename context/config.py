"""上下文构建配置。"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MEMORY_SEARCH_TYPES: tuple[str, ...] = ("working", "episodic", "semantic")


@dataclass(frozen=True)
class ContextConfig:
    """上下文构建配置。

    Attributes:
        max_tokens: 上下文总 token 上限
        reserve_ratio: 为固定分区（Role/Task/State/Output）预留的比例 (0.0-1.0)
        min_relevance: 最低相关性阈值，低于此值的 packet 直接丢弃
        enable_compression: 超 max_tokens 时是否启用兜底压缩
        recency_weight: 新近性在综合分中的权重 (0.0-1.0)
        relevance_weight: 相关性在综合分中的权重 (0.0-1.0)
        memory_search_types: Gather 时检索的记忆类型
        memory_limit_per_type: 每种记忆类型的检索条数上限
        rag_top_k: RAG 检索片段数量
        rag_enabled: 是否启用 RAG 采集
        recency_half_life_seconds: 新近性衰减半衰期（秒），默认 24 小时
        default_output_requirements: [Output] 分区默认文案
    """

    max_tokens: int = 8192
    reserve_ratio: float = 0.2
    min_relevance: float = 0.0
    enable_compression: bool = True
    recency_weight: float = 0.5
    relevance_weight: float = 0.5
    memory_search_types: tuple[str, ...] = DEFAULT_MEMORY_SEARCH_TYPES
    memory_limit_per_type: int = 3
    rag_top_k: int = 5
    rag_enabled: bool = True
    recency_half_life_seconds: float = 86_400.0
    default_output_requirements: str = "请根据以上信息回答用户问题。"

    def __post_init__(self) -> None:
        # frozen dataclass 需通过 object.__setattr__ 做校验与归一化
        object.__setattr__(
            self, "reserve_ratio", max(0.0, min(1.0, self.reserve_ratio))
        )
        object.__setattr__(
            self, "min_relevance", max(0.0, min(1.0, self.min_relevance))
        )
        object.__setattr__(
            self, "recency_weight", max(0.0, min(1.0, self.recency_weight))
        )
        object.__setattr__(
            self, "relevance_weight", max(0.0, min(1.0, self.relevance_weight))
        )
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if self.memory_limit_per_type <= 0:
            raise ValueError("memory_limit_per_type 必须大于 0")
        if self.rag_top_k <= 0:
            raise ValueError("rag_top_k 必须大于 0")
        if self.recency_half_life_seconds <= 0:
            raise ValueError("recency_half_life_seconds 必须大于 0")
        object.__setattr__(
            self,
            "memory_search_types",
            tuple(self.memory_search_types),
        )

    @property
    def reserved_tokens(self) -> int:
        """固定分区可使用的 token 预算（按 reserve_ratio 计算）。"""
        return int(self.max_tokens * self.reserve_ratio)

    @property
    def selectable_tokens(self) -> int:
        """Evidence + Context 可竞争使用的 token 预算。"""
        return self.max_tokens - self.reserved_tokens
