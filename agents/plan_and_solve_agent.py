"""Plan-and-Solve Agent：规划 -> 逐步求解 -> 汇总。"""

import ast
import re
from typing import TYPE_CHECKING, Any, Optional

from core.agent import Agent
from core.config import Config
from core.llm import BaseLLM
from core.message import Message
from memory.protocols import MemoryServiceProtocol

from .tool_loop import invoke_with_tool_registry
from prompts import (
    PLAN_AND_SOLVE_PLANNER_PROMPT,
    PLAN_AND_SOLVE_SOLVER_PROMPT,
    PLAN_AND_SOLVE_SYNTHESIS_PROMPT,
    render_prompt,
)

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

STEP_FAILED_MARKER = "[此步未能完成]"
PLAN_PARSE_ERROR_MESSAGE = "抱歉，我无法为这个问题制定有效的计划。"


class PlanParseError(Exception):
    """Planner 输出无法解析为有效非空步骤列表。"""


def parse_plan(text: str | None) -> list[str]:
    """将 Planner 输出解析为步骤字符串列表；失败返回空列表。"""
    raw = (text or "").strip()
    if not raw:
        return []

    for candidate in (raw, _extract_bracket_segment(raw)):
        if candidate is None:
            continue
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(value, list) or len(value) == 0:
            continue
        return [str(item) for item in value]

    return []


def _extract_bracket_segment(text: str) -> str | None:
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match is None:
        return None
    return match.group(0)


class PlanAndSolveAgent(Agent):
    """Plan-and-Solve：先规划步骤列表，再逐步求解，最后汇总。"""

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        planner_prompt: str = PLAN_AND_SOLVE_PLANNER_PROMPT,
        max_plan_retries: int = 3,
        verbose: bool = False,
        tool_registry: Optional["ToolRegistry"] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3,
        memory_service: MemoryServiceProtocol | None = None,
        enable_memory: bool = False,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.planner_prompt = planner_prompt
        self.max_plan_retries = max_plan_retries
        self.verbose = verbose
        self.plan_trace: list[str] = []
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.max_tool_iterations = max_tool_iterations
        self.memory_service = memory_service
        self.enable_memory = enable_memory and memory_service is not None

    @classmethod
    def with_agent_tools(
        cls,
        name: str,
        llm: BaseLLM,
        *,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        enable_search: bool = True,
        enable_calculator: bool = True,
        enable_memory: bool = False,
        enable_rag: bool = True,
        max_plan_retries: int = 3,
        max_tool_iterations: int = 3,
        memory_types: list[str] | None = None,
        memory_user_id: str = "default_user",
        memory_service: MemoryServiceProtocol | None = None,
        verbose: bool = False,
    ) -> "PlanAndSolveAgent":
        """使用默认工具集（含可选 memory / rag）创建 PlanAndSolveAgent。"""
        from memory.service import MemoryService
        from tools.agent_registry import create_agent_tool_registry

        service = memory_service
        if service is None and enable_memory:
            service = MemoryService(user_id=memory_user_id, memory_types=memory_types)

        registry = create_agent_tool_registry(
            enable_search=enable_search,
            enable_calculator=enable_calculator,
            enable_memory=enable_memory,
            enable_rag=enable_rag,
            memory_types=memory_types,
            memory_user_id=memory_user_id,
            memory_service=service,
        )
        return cls(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            max_plan_retries=max_plan_retries,
            verbose=verbose,
            tool_registry=registry,
            max_tool_iterations=max_tool_iterations,
            memory_service=service,
            enable_memory=enable_memory,
        )

    def run(self, input_text: str, **kwargs: Any) -> str:
        self.plan_trace = []
        temperature = kwargs.pop("temperature", self.config.temperature)

        print(f"\n📋 {self.name} 开始处理问题: {input_text}")
        episodic_context = self._retrieve_past_plans(input_text)
        try:
            plan = self._make_plan(input_text, temperature, episodic_context=episodic_context, **kwargs)
        except PlanParseError:
            self._save_final_answer(input_text, PLAN_PARSE_ERROR_MESSAGE)
            print(f"❌ {self.name} 无法制定有效计划")
            return PLAN_PARSE_ERROR_MESSAGE

        self.plan_trace.append(f"Plan: {plan}")
        self._log_llm_response("计划", str(plan))

        step_results: list[tuple[str, str]] = []
        history_text = "无"

        for index, step in enumerate(plan, start=1):
            print(f"\n--- 执行第 {index}/{len(plan)} 步 ---")
            result = self._solve_step(
                input_text, plan, step, history_text, temperature, **kwargs
            )
            step_results.append((step, result))
            self.plan_trace.append(f"Step {index}: {step} => {result}")
            self._log_llm_response(f"第 {index} 步", result)
            history_text = self._format_history(step_results)

        final_answer = self._synthesize(
            input_text, step_results, temperature, **kwargs
        )
        self._log_llm_response("汇总", final_answer)
        self._save_final_answer(input_text, final_answer)
        self._maybe_record_plan(input_text, plan, final_answer)
        print(f"✅ {self.name} Plan-and-Solve 完成")
        return final_answer

    def _make_plan(
        self,
        question: str,
        temperature: float,
        *,
        episodic_context: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.planner_prompt},
            {"role": "user", "content": question},
        ]
        if episodic_context:
            messages.append(
                {"role": "system", "content": f"历史相似计划参考:\n{episodic_context}"}
            )
        attempts = self.max_plan_retries + 1
        for attempt in range(1, attempts + 1):
            raw = self.llm.invoke(messages, temperature=temperature, **kwargs)
            plan = parse_plan(raw)
            if plan:
                return plan
            print(f"⚠️ 计划解析失败，重试 {attempt}/{attempts}")
        raise PlanParseError("planner output is not a valid non-empty list")

    def _solve_step(
        self,
        question: str,
        plan: list[str],
        current_step: str,
        history: str,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        prompt = render_prompt(
            PLAN_AND_SOLVE_SOLVER_PROMPT,
            question=question,
            plan=plan,
            current_step=current_step,
            history=history,
        )
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        try:
            if self.enable_tool_calling and self.tool_registry is not None:
                result = invoke_with_tool_registry(
                    self.llm,
                    self.tool_registry,
                    messages,
                    max_iterations=self.max_tool_iterations,
                    temperature=temperature,
                    **kwargs,
                )
            else:
                result = self.llm.invoke(messages, temperature=temperature, **kwargs)
        except Exception as e:
            print(f"❌ 步骤执行异常: {e}")
            return STEP_FAILED_MARKER
        if not result or not result.strip():
            return STEP_FAILED_MARKER
        return result

    def _synthesize(
        self,
        question: str,
        step_results: list[tuple[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> str:
        steps_and_results = self._format_history(step_results)
        prompt = render_prompt(
            PLAN_AND_SOLVE_SYNTHESIS_PROMPT,
            question=question,
            steps_and_results=steps_and_results,
        )
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, temperature=temperature, **kwargs) or ""

    def _retrieve_past_plans(self, question: str) -> str | None:
        """从 episodic 记忆检索相似历史计划（可选）。"""
        if not self.enable_memory or self.memory_service is None:
            return None
        try:
            results = self.memory_service.search(
                question,
                memory_type="episodic",
                limit=3,
            )
        except Exception as exc:
            print(f"⚠️ 历史计划检索失败: {exc}")
            return None
        if not results:
            return None
        lines: list[str] = []
        for result in results:
            content = result.get("content", "") if isinstance(result, dict) else str(result)
            lines.append(f"- {content}")
        return "\n".join(lines) if lines else None

    def _maybe_record_plan(
        self,
        question: str,
        plan: list[str],
        final_answer: str,
    ) -> None:
        """将完成的计划存入 episodic 记忆（可选）。"""
        if not self.enable_memory or self.memory_service is None:
            return
        content_text = (
            f"问题: {question}\n"
            f"计划: {plan}\n"
            f"最终答案: {final_answer}"
        )
        try:
            self.memory_service.add(
                content=content_text,
                memory_type="episodic",
                importance=0.6,
                metadata={
                    "agent_type": "plan_and_solve",
                    "question": question,
                    "plan_steps": len(plan),
                    "source": "plan_solve_hook",
                },
            )
        except Exception as exc:
            print(f"⚠️ 计划记忆写入失败: {exc}")

    @staticmethod
    def _format_history(step_results: list[tuple[str, str]]) -> str:
        if not step_results:
            return "无"
        lines = [
            f"- 步骤：{step}\n  结果：{result}"
            for step, result in step_results
        ]
        return "\n".join(lines)

    def _log_llm_response(self, label: str, content: str) -> None:
        if not self.verbose:
            return
        print(f"\n{'─' * 60}")
        print(f"📤 [{label}] 模型完整回复：")
        print(f"{'─' * 60}")
        print(content if content else "(空)")
        print(f"{'─' * 60}")

    def _save_final_answer(self, input_text: str, final_answer: str) -> None:
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_answer, role="assistant"))
