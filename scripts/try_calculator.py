"""CalculatorTool 试用脚本。

在项目根目录运行：

    python scripts/try_calculator.py
    python scripts/try_calculator.py --agent
    python scripts/try_calculator.py --task "请帮我计算 (1+2)*3 等于多少"

说明：当前计算器不支持 sqrt()，可用 16**0.5 表示平方根。
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
from tools.builtin.calculator import CalculatorTool
from tools.registry import ToolRegistry


def create_calculator_registry() -> ToolRegistry:
    """创建只包含 calculator 工具的注册表。"""
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    return registry


def test_calculator_tool() -> None:
    """直接通过 ToolRegistry 调用计算器。"""
    registry = create_calculator_registry()

    print("🧪 测试 CalculatorTool（经 ToolRegistry）\n")
    print(f"已注册工具: {registry.list_tools()}\n")

    test_cases = [
        "2 + 3",
        "10 - 4",
        "5 * 6",
        "15 / 3",
        "16**0.5",  # 平方根（当前实现不支持 sqrt()，用幂代替）
        "(16**0.5) + 2 * 3",
    ]

    for i, expression in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute("calculator", {"expression": expression})
        print(f"结果: {result}\n")


def test_with_simple_agent(task: str) -> None:
    """通过 SimpleAgent + Function Calling 让模型决定是否调用计算器。"""
    print("🤖 SimpleAgent + calculator 集成测试\n")

    llm = BaseLLM()
    registry = create_calculator_registry()
    agent = SimpleAgent(
        name="计算助手",
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
    if agent.has_tools():
        print(f"\n可用工具: {agent.list_tools()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="试用 CalculatorTool")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="只跑 SimpleAgent 集成测试（默认先跑注册表直调，再跑 Agent）",
    )
    parser.add_argument(
        "--registry-only",
        action="store_true",
        help="只跑 ToolRegistry 直调测试",
    )
    parser.add_argument(
        "--task",
        default="请帮我计算 (16**0.5) + 2 * 3 等于多少，并简要说明步骤",
        help="SimpleAgent 测试用的用户问题",
    )
    args = parser.parse_args()

    if args.agent:
        test_with_simple_agent(args.task)
        return

    if args.registry_only:
        test_calculator_tool()
        return

    test_calculator_tool()
    print("=" * 60 + "\n")
    test_with_simple_agent(args.task)


if __name__ == "__main__":
    main()
