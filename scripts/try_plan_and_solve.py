"""PlanAndSolveAgent 真机试用脚本。

在项目根目录运行（需已配置 .env 中的 LLM_* 变量）：

    python scripts/try_plan_and_solve.py
    python scripts/try_plan_and_solve.py --task "规划并回答：如何入门 Python"
    python scripts/try_plan_and_solve.py --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.plan_and_solve_agent import PlanAndSolveAgent
from core.llm import BaseLLM

DEFAULT_TASK = "请规划并回答：学习机器学习应该先掌握哪些基础知识？"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="试用 PlanAndSolveAgent（调用真实 LLM API）"
    )
    parser.add_argument("--task", default=None, help="自定义任务")
    parser.add_argument(
        "--max-plan-retries",
        type=int,
        default=3,
        help="计划解析失败时最多重试次数（默认 3）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印每次 LLM 的完整回复",
    )
    args = parser.parse_args()
    verbose = not args.quiet
    task = args.task or DEFAULT_TASK

    print("正在初始化 LLM（读取 .env）...")
    llm = BaseLLM()
    agent = PlanAndSolveAgent(
        name="我的规划求解助手",
        llm=llm,
        max_plan_retries=args.max_plan_retries,
        verbose=verbose,
    )

    print(f"\n任务: {task}\n")
    result = agent.run(task)
    print(f"\n{'=' * 60}")
    print(f"最终结果:\n{result}")
    print(f"{'=' * 60}")
    print("\n执行轨迹:")
    for i, line in enumerate(agent.plan_trace, start=1):
        print(f"  {i}. {line}")


if __name__ == "__main__":
    main()
