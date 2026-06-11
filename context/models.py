"""上下文领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# 分区标识，写入 ContextPacket.metadata["section"]，供 template / selector 使用
SECTION_ROLE = "role"
SECTION_TASK = "task"
SECTION_STATE = "state"
SECTION_EVIDENCE = "evidence"
SECTION_CONTEXT = "context"
SECTION_OUTPUT = "output"


@dataclass
class ContextPacket:
    """候选信息包，Gather 阶段的统一数据单元。

    每条来自历史、记忆或 RAG 的信息都被封装为一个 packet，
    携带内容、时间戳、token 数和相关性分数，供 Select 阶段统一排序选取。

    Attributes:
        content: 信息内容
        timestamp: 时间戳，用于新近性评分
        token_count: Token 数量，用于预算装箱
        relevance_score: 相关性分数 (0.0-1.0)
        metadata: 扩展元数据，必含 section（目标分区）与 source（来源）
    """

    content: str
    timestamp: datetime
    token_count: int
    relevance_score: float = 0.5
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class BuiltContext:
    """构建完成的上下文，ContextBuilder.build() 的返回值。

    Attributes:
        text: 六分区完整文本
        messages: 可直接传给 LLM 的消息列表，默认单条 system 消息
        stats: 调试统计（token 预算、选中/丢弃 packet 数等）
    """

    text: str
    messages: list[dict[str, str]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
