"""SearchTool 试用脚本。

在项目根目录运行：

    python scripts/try_search.py
    python scripts/try_search.py --mode tavily --query "最新 AI 新闻"
    python scripts/try_search.py --mode hybrid --routing llm --query "python 列表用法"
    python scripts/try_search.py --agent --task "查一下今天的天气并总结"

说明：需要在 .env 中配置 TAVILY_API_KEY 和/或 SERPAPI_API_KEY。
未配置密钥时工具会返回中文错误提示，不会抛出异常。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM
from tools.builtin.search import SearchTool
from tools.registry import ToolRegistry


def create_search_registry(mode: str, routing: str) -> ToolRegistry:
    """创建只包含 search 工具的注册表。"""
    registry = ToolRegistry()
    llm = BaseLLM() if routing == "llm" else None
    registry.register_tool(SearchTool(mode=mode, routing=routing, llm=llm))
    return registry


def test_search_tool(mode: str, routing: str, query: str) -> None:
    """直接通过 ToolRegistry 调用搜索工具。"""
    registry = create_search_registry(mode, routing)

    print(f"🧪 测试 SearchTool（mode={mode}, routing={routing}，经 ToolRegistry）\n")
    print(f"已注册工具: {registry.list_tools()}\n")
    print(f"查询: {query}")
    result = registry.execute("search", {"query": query})
    print(f"结果:\n{result}\n")


def test_with_simple_agent(mode: str, routing: str, task: str) -> None:
    """通过 SimpleAgent + Function Calling 让模型决定是否调用搜索。"""
    print("🤖 SimpleAgent + search 集成测试\n")

    llm = BaseLLM()
    registry = create_search_registry(mode, routing)
    agent = SimpleAgent(
        name="搜索助手",
        llm=llm,
        tool_registry=registry,
        enable_tool_calling=True,
        max_tool_iterations=3,
    )

    print(f"用户问题: {task}\n")
    print("--- Agent 运行中（可能多次调用 LLM / 工具）---\n")
    answer = agent.run(task)
    print(f"\n{'=' * 60}")
    print(f"最终回答:\n{answer}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="试用 SearchTool")
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "tavily", "serpapi"],
        help="搜索模式，默认 hybrid",
    )
    parser.add_argument(
        "--query",
        default="什么是 ReAct Agent",
        help="ToolRegistry 直调测试用的查询",
    )
    parser.add_argument(
        "--routing",
        default="keyword",
        choices=["keyword", "llm"],
        help="hybrid 路由方式：keyword/llm（llm 会额外调用一次 LLM）",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="跑 SimpleAgent 集成测试",
    )
    parser.add_argument(
        "--task",
        default="帮我搜索一下什么是 ReAct Agent，并简要总结",
        help="SimpleAgent 测试用的用户问题",
    )
    args = parser.parse_args()

    if args.agent:
        test_with_simple_agent(args.mode, args.routing, args.task)
        return

    test_search_tool(args.mode, args.routing, args.query)


if __name__ == "__main__":
    main()
