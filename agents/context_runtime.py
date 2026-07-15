"""ContextBuilder 与 Agent 运行时的衔接辅助。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from context import BuiltContext, ContextBuilder
from memory.hooks import MemoryHookConfig

if TYPE_CHECKING:
    from tools.builtin.memory_tool import MemoryTool


def disable_legacy_memory_retrieve(memory_hooks: MemoryHookConfig | None) -> None:
    """ContextBuilder 已负责 Gather 记忆，关闭旧的 hook 检索避免重复。"""
    if memory_hooks is not None:
        memory_hooks.retrieve_before_run = False


def resolve_session_id(
    memory_hooks: MemoryHookConfig | None,
    memory_tool: MemoryTool | None,
) -> str | None:
    """解析记忆检索使用的 session_id。"""
    if memory_hooks is not None and memory_hooks.session_id:
        return memory_hooks.session_id
    if memory_tool is not None and memory_tool.current_session_id:
        return memory_tool.current_session_id
    return None


def build_context_messages(
    context_builder: ContextBuilder,
    *,
    input_text: str,
    system_prompt: str | None,
    conversation_history: list,
    session_id: str | None = None,
    state: str | None = None,
    output_requirements: str | None = None,
) -> BuiltContext:
    """调用 ContextBuilder 并返回 BuiltContext。"""
    return context_builder.build(
        user_query=input_text,
        conversation_history=conversation_history,
        system_instructions=system_prompt,
        session_id=session_id,
        state=state,
        output_requirements=output_requirements,
    )
