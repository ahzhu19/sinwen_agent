"""工具注册表"""
from typing import Any

from .base import Tool


class ToolRegistry:
    """管理 Agent 可用工具及其 OpenAI schema"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True) -> None:
        """注册工具。auto_expand 预留扩展，当前直接注册单个工具。"""
        _ = auto_expand
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> bool:
        if name not in self._tools:
            return False
        del self._tools[name]
        return True

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def describe_tools(self) -> str:
        """返回适合放进 ReAct 提示词的工具说明。"""
        if not self._tools:
            return "无可用工具"

        descriptions: list[str] = []
        for tool in self._tools.values():
            descriptions.append(
                f"- {tool.name}: {tool.description}. 参数 schema: {tool.parameters()}"
            )
        return "\n".join(descriptions)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未找到工具 '{name}'"
        try:
            return tool.run(**arguments)
        except Exception as e:
            return f"错误：工具 '{name}' 执行失败 - {e}"

    async def aexecute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未找到工具 '{name}'"
        try:
            return await tool.arun(**arguments)
        except Exception as e:
            return f"错误：工具 '{name}' 执行失败 - {e}"
