"""LLM 工具调用相关类型"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """模型发起的一次工具调用"""

    id: str
    name: str
    arguments: str  # JSON 字符串


@dataclass
class LLMToolResponse:
    """带工具调用信息的 LLM 响应"""

    content: str | None
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
