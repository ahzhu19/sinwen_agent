"""ReflectionAgent 真机试用脚本。

在项目根目录运行（需已配置 .env 中的 LLM_* 变量）：

    python scripts/try_reflection.py
    python scripts/try_reflection.py --mode code
    python scripts/try_reflection.py --task "写一句关于春天的诗"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 保证从项目根执行 `python scripts/try_reflection.py` 时能导入 agents、core
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.reflection_agent import ReflectionAgent
from core.llm import BaseLLM

# 通用反思助手：使用 prompts.py 中的默认初稿 / 审稿 / 改写模板
DEFAULT_TASK = "写一篇关于通讯领域物理层安全的技术发展文字（400 字）"

# 代码场景：仅自定义「初稿」系统提示；审稿与改写仍用默认模板
CODE_SYSTEM_PROMPT = """你是 Python 专家。
请针对用户任务编写完整、可运行的代码，并附上一句简要说明。"""

CODE_TASK = "编写一个函数 fibonacci(n)，返回第 n 个斐波那契数（n 从 0 开始）"


def run_agent(agent: ReflectionAgent, task: str) -> str:
    result = agent.run(task)
    print(f"\n{'=' * 60}")
    print(f"最终结果:\n{result}")
    print(f"{'=' * 60}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="试用 ReflectionAgent（调用真实 LLM API）")
    parser.add_argument(
        "--mode",
        choices=("general", "code"),
        default="general",
        help="general=写文章；code=写 Python 代码（默认 general）",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="自定义任务文本；不填则使用各 mode 的默认示例",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="最多反思轮数（默认 2，每轮可能多调几次 API）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印每次 LLM 的完整回复（默认会打印）",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    print("正在初始化 LLM（读取 .env）...")
    llm = BaseLLM()

    if args.mode == "general":
        task = args.task or DEFAULT_TASK
        agent = ReflectionAgent(
            name="我的反思助手",
            llm=llm,
            max_iterations=args.max_iterations,
            verbose=verbose,
        )
    else:
        task = args.task or CODE_TASK
        agent = ReflectionAgent(
            name="我的代码生成助手",
            llm=llm,
            system_prompt=CODE_SYSTEM_PROMPT,
            max_iterations=args.max_iterations,
            verbose=verbose,
        )

    print(f"\n模式: {args.mode} | 最大反思轮数: {args.max_iterations}")
    print(f"任务: {task}\n")
    run_agent(agent, task)


if __name__ == "__main__":
    main()
