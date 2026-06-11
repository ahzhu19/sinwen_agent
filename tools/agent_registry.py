"""为 Agent 组装默认工具注册表。"""

from __future__ import annotations

from memory.protocols import MemoryServiceProtocol
from tools.builtin.calculator import CalculatorTool
from tools.builtin.memory_tool import MemoryTool
from tools.builtin.rag_tool import RagTool
from tools.builtin.search import SearchTool
from tools.registry import ToolRegistry


def create_agent_tool_registry(
    *,
    enable_search: bool = True,
    enable_calculator: bool = True,
    enable_memory: bool = False,
    enable_rag: bool = True,
    search_mode: str = "hybrid",
    search_routing: str = "keyword",
    memory_types: list[str] | None = None,
    memory_user_id: str = "default_user",
    memory_service: MemoryServiceProtocol | None = None,
) -> ToolRegistry:
    """创建 Agent 可用的工具注册表。

    默认启用 search、calculator、rag；memory 需显式开启（依赖数据库与 embedding）。
    """
    registry = ToolRegistry()

    if enable_search:
        registry.register_tool(SearchTool(mode=search_mode, routing=search_routing))
    if enable_calculator:
        registry.register_tool(CalculatorTool())
    if enable_memory:
        types = memory_types or ["working"]
        registry.register_tool(
            MemoryTool(
                user_id=memory_user_id,
                memory_types=types,
                memory_service=memory_service,
            ),
        )
    if enable_rag:
        registry.register_tool(RagTool())

    return registry
