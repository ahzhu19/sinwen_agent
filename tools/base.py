"""工具基类"""
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """可被 Agent 通过 Function Calling 调用的工具"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（唯一）"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，供模型理解用途"""

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """参数的 JSON Schema（OpenAI function parameters）"""

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI tools API 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters(),
            },
        }

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """执行工具，返回字符串结果"""

    async def arun(self, **kwargs: Any) -> str:
        """异步执行工具（默认通过线程池卸载同步 run）。"""
        import asyncio

        return await asyncio.to_thread(self.run, **kwargs)
