"""简单 Agent 实现 - 基础对话与可选工具调用"""

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

from core.agent import Agent
from core.config import Config
from core.llm import BaseLLM
from core.message import Message
from .prompts import DEFAULT_SIMPLE_AGENT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from tools.base import Tool
    from tools.registry import ToolRegistry


class SimpleAgent(Agent):
    """简单的对话 Agent。

    特性：
    - 纯对话模式（无工具）
    - 可选 Function Calling 工具调用
    - 同步流式输出（stream_run，不含工具循环）
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: Optional[str] = DEFAULT_SIMPLE_AGENT_SYSTEM_PROMPT,
        config: Optional[Config] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        enable_tool_calling: bool = True,
        max_tool_iterations: int = 3,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        self.max_tool_iterations = max_tool_iterations

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Agent，返回完整回复。"""
        if not self.enable_tool_calling or self.tool_registry is None:
            return self._run_chat(input_text, **kwargs)
        return self._run_with_tools(input_text, **kwargs)

    def _run_chat(self, input_text: str, **kwargs: Any) -> str:
        messages = self._build_messages(input_text)
        temperature = kwargs.pop("temperature", self.config.temperature)

        response_text = self.llm.invoke(messages, temperature=temperature, **kwargs)
        if response_text is None:
            response_text = ""

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=response_text, role="assistant"))
        return response_text

    def _run_with_tools(self, input_text: str, **kwargs: Any) -> str:
        messages = self._build_messages(input_text)
        temperature = kwargs.pop("temperature", self.config.temperature)
        tool_schemas = self.tool_registry.get_tool_schemas()  # type: ignore[union-attr]

        current_iteration = 0
        final_response = ""

        while current_iteration < self.max_tool_iterations:
            current_iteration += 1
            try:
                response = self.llm.invoke_with_tools(
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    temperature=temperature,
                    **kwargs,
                )
            except Exception as e:
                print(f"❌ LLM 调用失败: {e}")
                break

            tool_calls = response.tool_calls
            if not tool_calls:
                final_response = response.content or "抱歉，我无法回答这个问题。"
                break

            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.arguments)
                except json.JSONDecodeError as e:
                    print(f"❌ 工具参数解析失败: {e}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"错误：参数格式不正确 - {e}",
                    })
                    continue

                result = self._execute_tool_call(tool_call.name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        if current_iteration >= self.max_tool_iterations and not final_response:
            fallback = self.llm.invoke(messages, temperature=temperature, **kwargs)
            final_response = fallback or ""

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=final_response, role="assistant"))
        return final_response

    def _execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if self.tool_registry is None:
            return "错误：未配置工具注册表"
        return self.tool_registry.execute(tool_name, arguments)

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """流式运行 Agent（纯对话，不执行工具循环）。"""
        messages = self._build_messages(input_text)
        temperature = kwargs.pop("temperature", self.config.temperature)

        full_response = ""
        for chunk in self.llm.stream_invoke(messages, temperature=temperature, **kwargs):
            full_response += chunk
            yield chunk

        self.add_message(Message(content=input_text, role="user"))
        self.add_message(Message(content=full_response, role="assistant"))

    def add_tool(self, tool: "Tool", auto_expand: bool = True) -> None:
        """添加工具；若尚无注册表则自动创建。"""
        if self.tool_registry is None:
            from tools.registry import ToolRegistry

            self.tool_registry = ToolRegistry()
            self.enable_tool_calling = True
        self.tool_registry.register_tool(tool, auto_expand=auto_expand)

    def remove_tool(self, tool_name: str) -> bool:
        if self.tool_registry is None:
            return False
        return self.tool_registry.unregister_tool(tool_name)

    def list_tools(self) -> list[str]:
        if self.tool_registry is None:
            return []
        return self.tool_registry.list_tools()

    def has_tools(self) -> bool:
        return self.enable_tool_calling and self.tool_registry is not None

    def _build_messages(self, input_text: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append(msg.to_dict())

        messages.append({"role": "user", "content": input_text})
        return messages
