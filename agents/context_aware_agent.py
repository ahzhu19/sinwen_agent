"""具有上下文感知能力的 SimpleAgent 子类。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

from context import BuiltContext, ContextBuilder, ContextConfig
from core.config import Config
from core.llm import BaseLLM
from memory.hooks import MemoryHookConfig
from memory.protocols import MemoryServiceProtocol
from prompts import DEFAULT_SIMPLE_AGENT_SYSTEM_PROMPT

from .context_runtime import (
    build_context_messages,
    disable_legacy_memory_retrieve,
    resolve_session_id,
)
from .memory_runtime import (
    build_memory_hook_config,
    resolve_memory_hooks_enabled,
)
from .simple_agent import SimpleAgent

if TYPE_CHECKING:
    from tools.base import Tool
    from tools.builtin.memory_tool import MemoryTool
    from tools.builtin.rag_tool import RagTool
    from tools.registry import ToolRegistry


class ContextAwareAgent(SimpleAgent):
    """具有上下文感知能力的 Agent。

    在 SimpleAgent 基础上，用 ContextBuilder 统一组装六分区上下文：
    Gather（历史 + 记忆 + RAG）→ Select → Structure → Compress。

    启用 ContextBuilder 时自动关闭 memory_hooks 的 run 前检索，
    避免与 Gather 重复；run 后自动记录（record_after_run）仍保留。
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
        memory_service: MemoryServiceProtocol | None = None,
        memory_hooks: MemoryHookConfig | None = None,
        context_builder: ContextBuilder | None = None,
        context_config: ContextConfig | None = None,
        enable_context: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry,
            enable_tool_calling=enable_tool_calling,
            max_tool_iterations=max_tool_iterations,
            memory_service=memory_service,
            memory_hooks=memory_hooks,
        )
        self.context_config = context_config or ContextConfig()
        self.enable_context = enable_context
        self._last_built_context: BuiltContext | None = None
        self._context_memory_tool: MemoryTool | None = None

        if context_builder is not None:
            self.context_builder = context_builder
        elif enable_context:
            self.context_builder = ContextBuilder(config=self.context_config)
        else:
            self.context_builder = None

        if self.context_builder is not None and enable_context:
            disable_legacy_memory_retrieve(self.memory_hooks)

    @property
    def last_built_context(self) -> BuiltContext | None:
        """最近一次 run / stream_run 构建的上下文，便于调试与评估。"""
        return self._last_built_context

    @classmethod
    def with_agent_tools(
        cls,
        name: str,
        llm: BaseLLM,
        *,
        system_prompt: Optional[str] = DEFAULT_SIMPLE_AGENT_SYSTEM_PROMPT,
        config: Optional[Config] = None,
        enable_search: bool = True,
        enable_calculator: bool = True,
        enable_memory: bool = False,
        enable_rag: bool = True,
        max_tool_iterations: int = 5,
        memory_types: list[str] | None = None,
        enable_memory_hooks: bool | None = None,
        memory_user_id: str = "default_user",
        memory_hook_session_id: str | None = None,
        memory_service: MemoryServiceProtocol | None = None,
        context_config: ContextConfig | None = None,
        enable_context: bool = True,
    ) -> "ContextAwareAgent":
        """使用默认工具集创建 ContextAwareAgent，并装配 ContextBuilder。"""
        from memory.service import MemoryService
        from tools.agent_registry import create_agent_tool_registry
        from tools.builtin.memory_tool import MemoryTool
        from tools.builtin.rag_tool import RagTool

        hooks_enabled = resolve_memory_hooks_enabled(enable_memory, enable_memory_hooks)
        service = memory_service
        if service is None and (enable_memory or hooks_enabled):
            service = MemoryService(user_id=memory_user_id, memory_types=memory_types)

        hooks = build_memory_hook_config(memory_hook_session_id) if hooks_enabled else None

        registry = create_agent_tool_registry(
            enable_search=enable_search,
            enable_calculator=enable_calculator,
            enable_memory=enable_memory,
            enable_rag=enable_rag,
            memory_types=memory_types,
            memory_user_id=memory_user_id,
            memory_service=service,
        )

        memory_tool = registry.get_tool("memory")
        if not isinstance(memory_tool, MemoryTool) and service is not None and enable_context:
            memory_tool = MemoryTool(
                user_id=memory_user_id,
                session_id=memory_hook_session_id,
                memory_types=memory_types or ["working"],
                memory_service=service,
            )

        rag_tool = registry.get_tool("rag")
        rag_tool = rag_tool if isinstance(rag_tool, RagTool) else None

        context_builder = None
        if enable_context:
            context_builder = ContextBuilder(
                memory_tool=memory_tool if isinstance(memory_tool, MemoryTool) else None,
                rag_tool=rag_tool,
                config=context_config or ContextConfig(),
            )

        agent = cls(
            name=name,
            llm=llm,
            system_prompt=system_prompt,
            config=config,
            tool_registry=registry,
            enable_tool_calling=True,
            max_tool_iterations=max_tool_iterations,
            memory_service=service,
            memory_hooks=hooks,
            context_builder=context_builder,
            context_config=context_config,
            enable_context=enable_context,
        )
        agent._context_memory_tool = (
            memory_tool if isinstance(memory_tool, MemoryTool) else None
        )
        return agent

    def _build_messages(
        self,
        input_text: str,
        *,
        memory_context: str | None = None,
        state: str | None = None,
        output_requirements: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        _ = kwargs
        if not self.enable_context or self.context_builder is None:
            return super()._build_messages(input_text, memory_context=memory_context)

        built = build_context_messages(
            self.context_builder,
            input_text=input_text,
            system_prompt=self.system_prompt,
            conversation_history=self.get_history(),
            session_id=resolve_session_id(
                self.memory_hooks,
                self._context_memory_tool,
            ),
            state=state,
            output_requirements=output_requirements,
        )
        self._last_built_context = built

        messages: list[dict[str, Any]] = list(built.messages)
        messages.append({"role": "user", "content": input_text})
        return messages

    def stream_run(
        self,
        input_text: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式运行；上下文构建逻辑与 run() 共用 _build_messages。"""
        if not self.enable_tool_calling or self.tool_registry is None:
            yield from self._stream_chat(input_text, memory_context=None, **kwargs)
            return
        yield from self._stream_with_tools(input_text, memory_context=None, **kwargs)
