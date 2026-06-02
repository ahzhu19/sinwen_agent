"""文本 ReAct Agent 实现。"""

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from core.agent import Agent
from core.config import Config
from core.llm import BaseLLM
from core.message import Message

from .prompts import DEFAULT_REACT_SYSTEM_PROMPT, REACT_USER_PROMPT_TEMPLATE, render_prompt

if TYPE_CHECKING:
    from tools.registry import ToolRegistry


@dataclass
class ReActStep:
    """一次 ReAct 模型输出的解析结果。"""

    thought: str
    action_name: str | None
    action_input: Any
    raw: str

    @property
    def is_finish(self) -> bool:
        return self.action_name == "Finish"


def parse_react_output(output: str | None) -> ReActStep:
    """解析模型输出中的 Thought、Action 和 Action Input。"""
    raw = output or ""
    thought = _extract_field(raw, "Thought") or ""
    action_name = _extract_field(raw, "Action")
    action_input_text = _extract_field(raw, "Action Input")

    action_input: Any = None
    if action_input_text is not None:
        action_input = _parse_action_input(action_input_text)

    return ReActStep(
        thought=thought,
        action_name=action_name,
        action_input=action_input,
        raw=raw,
    )


def _extract_field(text: str, field_name: str) -> str | None:
    pattern = rf"^{re.escape(field_name)}:\s*(.*?)(?=^\w+(?:\s+\w+)*:|\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_action_input(action_input: str) -> Any:
    try:
        return json.loads(action_input)
    except json.JSONDecodeError:
        return action_input


class ReActAgent(Agent):
    """基于文本 Thought/Action/Observation 循环的 ReAct Agent。"""

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tool_registry: "ToolRegistry",
        system_prompt: Optional[str] = DEFAULT_REACT_SYSTEM_PROMPT,
        user_prompt_template: str = REACT_USER_PROMPT_TEMPLATE,
        config: Optional[Config] = None,
        max_steps: int = 10,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.user_prompt_template = user_prompt_template
        self.max_steps = max_steps
        self.react_trace: list[str] = []

    def run(self, input_text: str, **kwargs: Any) -> str:
        """运行文本 ReAct 循环，返回最终答案。"""
        self.react_trace = []
        temperature = kwargs.pop("temperature", self.config.temperature)

        print(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        for current_step in range(1, self.max_steps + 1):
            print(f"\n--- 第 {current_step} 步 ---")
            messages = self._build_messages(input_text)
            response_text = self.llm.invoke(
                messages,
                temperature=temperature,
                **kwargs,
            )
            step = parse_react_output(response_text)

            if step.is_finish:
                final_answer = str(step.action_input or "")
                self._save_final_answer(input_text, final_answer)
                print(f"✅ {self.name} ReAct 响应完成")
                return final_answer

            if step.action_name:
                observation = self._execute_action(step)
                self._append_trace(step, observation)
                continue

            self.react_trace.append(
                "Observation: 输出格式无效，请按 Thought/Action/Action Input 格式重试。"
            )

        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self._save_final_answer(input_text, final_answer)
        return final_answer

    def _build_messages(self, input_text: str) -> list[dict[str, str]]:
        history = "\n".join(self.react_trace) if self.react_trace else "无"
        user_prompt = render_prompt(
            self.user_prompt_template,
            tools=self.tool_registry.describe_tools(),
            question=input_text,
            history=history,
        )
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _execute_action(self, step: ReActStep) -> str:
        if not isinstance(step.action_input, dict):
            return "错误：Action Input 必须是 JSON 对象"
        return self.tool_registry.execute(step.action_name or "", step.action_input)

    def _append_trace(self, step: ReActStep, observation: str) -> None:
        if step.thought:
            self.react_trace.append(f"Thought: {step.thought}")
        self.react_trace.append(f"Action: {step.action_name}")
        self.react_trace.append(f"Action Input: {step.action_input}")
        self.react_trace.append(f"Observation: {observation}")

    def _save_final_answer(self, input_text: str, final_answer: str) -> None:
        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_answer, role="assistant"))
