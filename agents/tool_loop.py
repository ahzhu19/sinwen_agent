"""非 ReAct Agent 的可选 Function Calling 工具循环。"""

from __future__ import annotations

import json
from typing import Any

from core.llm import BaseLLM
from tools.registry import ToolRegistry


def invoke_with_tool_registry(
    llm: BaseLLM,
    tool_registry: ToolRegistry,
    messages: list[dict[str, Any]],
    *,
    max_iterations: int = 3,
    temperature: float = 0,
    **kwargs: Any,
) -> str:
    """在已有 messages 上运行工具循环，返回最终文本。"""
    tool_schemas = tool_registry.get_tool_schemas()
    final_response = ""

    for _ in range(max_iterations):
        try:
            response = llm.invoke_with_tools(
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                temperature=temperature,
                **kwargs,
            )
        except Exception:
            break

        if not response.tool_calls:
            return response.content or final_response

        messages.append({
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in response.tool_calls
            ],
        })

        for tool_call in response.tool_calls:
            try:
                arguments = json.loads(tool_call.arguments)
            except json.JSONDecodeError as exc:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"错误：参数格式不正确 - {exc}",
                })
                continue

            result = tool_registry.execute(tool_call.name, arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    fallback = llm.invoke(messages, temperature=temperature, **kwargs)
    return fallback or final_response
