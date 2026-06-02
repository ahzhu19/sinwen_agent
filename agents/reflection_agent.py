"""反思 Agent 实现：生成 -> 自我批评 -> 改进 循环。"""

from typing import Any, Optional

from core.agent import Agent
from core.config import Config
from core.llm import BaseLLM
from core.message import Message

from .prompts import (
    REFLECTION_CRITIQUE_PROMPT,
    REFLECTION_INITIAL_SYSTEM_PROMPT,
    REFLECTION_NO_CHANGES_MARKER,
    REFLECTION_REVISE_PROMPT,
    render_prompt,
)


class ReflectionAgent(Agent):
    """通过自我批评迭代改进答案的 Agent。

    单个 LLM 通过不同提示词扮演执行者与审稿人，循环执行
    生成初稿 -> 反思批评 -> 改进，直到审稿人满意或达到 max_iterations。
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: Optional[str] = REFLECTION_INITIAL_SYSTEM_PROMPT,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.reflection_trace: list[str] = []

    def run(self, input_text: str, **kwargs: Any) -> str:
        """运行反思循环，返回最终答案。"""
        self.reflection_trace = []
        temperature = kwargs.pop("temperature", self.config.temperature)

        print(f"\n🪞 {self.name} 开始处理问题: {input_text}")
        answer = self._generate_initial(input_text, temperature, **kwargs)
        self._log_llm_response("初稿", answer)
        self.reflection_trace.append(f"Draft: {answer}")

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- 反思第 {iteration} 轮 ---")
            critique = self._reflect(input_text, answer, temperature, **kwargs)
            self._log_llm_response(f"第 {iteration} 轮 · 审稿", critique)
            self.reflection_trace.append(f"Critique: {critique}")

            if self._is_satisfied(critique):
                print("✅ 审稿人认为无需改进")
                break

            answer = self._revise(input_text, answer, critique, temperature, **kwargs)
            self._log_llm_response(f"第 {iteration} 轮 · 改写", answer)
            self.reflection_trace.append(f"Revised: {answer}")

        self._save_final_answer(input_text, answer)
        print(f"✅ {self.name} 反思完成")
        return answer

    def _generate_initial(
        self, input_text: str, temperature: float, **kwargs: Any
    ) -> str:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": input_text})
        return self.llm.invoke(messages, temperature=temperature, **kwargs) or ""

    def _reflect(
        self, input_text: str, answer: str, temperature: float, **kwargs: Any
    ) -> str:
        prompt = render_prompt(
            REFLECTION_CRITIQUE_PROMPT,
            marker=REFLECTION_NO_CHANGES_MARKER,
            question=input_text,
            answer=answer,
        )
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, temperature=temperature, **kwargs) or ""

    def _revise(
        self,
        input_text: str,
        answer: str,
        critique: str,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        prompt = render_prompt(
            REFLECTION_REVISE_PROMPT,
            question=input_text,
            answer=answer,
            critique=critique,
        )
        messages = [{"role": "user", "content": prompt}]
        return self.llm.invoke(messages, temperature=temperature, **kwargs) or ""

    def _is_satisfied(self, critique: str) -> bool:
        return REFLECTION_NO_CHANGES_MARKER in critique

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
