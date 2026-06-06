"""MemoryTool + Agent 真机试用脚本。

在项目根目录运行：

    # 仅 MemoryTool（默认 working，无需 Docker）
    uv run python scripts/try_memory_agent.py tool

    # SimpleAgent + memory（需 LLM API）
    uv run python scripts/try_memory_agent.py simple --task "记住我喜欢深色主题"

    # ReActAgent + memory（需 LLM API）
    uv run python scripts/try_memory_agent.py react --task "统计当前工作记忆条数"

    # 启用 episodic/semantic 需 Docker 与 .env 配置
    uv run python scripts/try_memory_agent.py tool --memory-types working,episodic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.react_agent import ReActAgent
from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM
from memory.config import MemoryConfig
from memory.manager import MemoryManager
from tools.agent_registry import create_agent_tool_registry
from tools.builtin.memory_tool import MemoryTool


def _parse_memory_types(raw: str | None) -> list[str]:
    if not raw:
        return ["working"]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _build_memory_tool(memory_types: list[str]) -> MemoryTool:
    config = MemoryConfig.from_env()
    manager = MemoryManager(
        config=config,
        user_id="try_memory_agent_user",
        enable_working="working" in memory_types,
        enable_episodic="episodic" in memory_types,
        enable_semantic="semantic" in memory_types,
        enable_perceptual="perceptual" in memory_types,
    )
    return MemoryTool(
        user_id="try_memory_agent_user",
        session_id="try_memory_agent_session",
        memory_manager=manager,
        memory_types=memory_types,
    )


def run_tool_demo(memory_types: list[str]) -> None:
    tool = _build_memory_tool(memory_types)
    session = tool.current_session_id or "(auto)"

    print(f"会话 ID: {session}")
    print("类型:", ", ".join(memory_types))

    steps = [
        ("add", {"content": "用户偏好深色主题", "importance": 0.85}),
        ("add", {"content": "临时草稿：待删除", "importance": 0.1}),
        ("search", {"query": "深色主题", "memory_type": "working"}),
        ("summary", {"limit": 5}),
        ("stats", {}),
        ("forget", {"importance_threshold": 0.5}),
        ("stats", {}),
    ]
    for action, payload in steps:
        print(f"\n>>> memory {action} {payload}")
        print(tool.execute(action, **payload))

    if "working" in memory_types and "episodic" in memory_types:
        print("\n>>> memory consolidate")
        print(tool.execute("consolidate", importance_threshold=0.5))
        print("\n>>> memory stats (after consolidate)")
        print(tool.execute("stats"))

    print("\n>>> memory clear_all")
    print(tool.execute("clear_all"))


def _create_llm() -> BaseLLM:
    return BaseLLM()


def run_simple_agent(task: str, memory_types: list[str]) -> None:
    llm = _create_llm()
    agent = SimpleAgent.with_agent_tools(
        name="记忆助手",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
    )
    memory_tool = _build_memory_tool(memory_types)
    if agent.tool_registry is not None:
        agent.tool_registry.unregister_tool("memory")
        agent.tool_registry.register_tool(memory_tool)

    print(f"任务: {task}\n")
    print(agent.run(task))


def run_react_agent(task: str, memory_types: list[str]) -> None:
    llm = _create_llm()
    agent = ReActAgent.with_agent_tools(
        name="ReAct记忆助手",
        llm=llm,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        memory_types=memory_types,
        max_steps=8,
    )
    memory_tool = _build_memory_tool(memory_types)
    agent.tool_registry.unregister_tool("memory")
    agent.tool_registry.register_tool(memory_tool)

    print(f"任务: {task}\n")
    print(agent.run(task))


def main() -> None:
    load_dotenv(_ROOT / ".env")
    parser = argparse.ArgumentParser(description="试用 MemoryTool 与 Agent 集成")
    parser.add_argument(
        "mode",
        choices=["tool", "simple", "react"],
        nargs="?",
        default="tool",
        help="运行模式：tool（默认）/ simple / react",
    )
    parser.add_argument("--task", default="帮我记住并统计当前会话的工作记忆")
    parser.add_argument(
        "--memory-types",
        default="working",
        help="逗号分隔，如 working 或 working,episodic,semantic",
    )
    args = parser.parse_args()
    memory_types = _parse_memory_types(args.memory_types)

    try:
        if args.mode == "tool":
            run_tool_demo(memory_types)
        elif args.mode == "simple":
            run_simple_agent(args.task, memory_types)
        else:
            run_react_agent(args.task, memory_types)
    except Exception as exc:
        print(f"\n❌ 试用失败：{exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
