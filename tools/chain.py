"""工具链：按步骤顺序执行多个工具。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .registry import ToolRegistry


@dataclass(frozen=True)
class ChainStep:
    tool_name: str
    arguments: dict[str, str]
    output_key: str


def _render_arguments(arguments: dict[str, str], context: dict[str, Any]) -> dict[str, Any] | str:
    rendered: dict[str, Any] = {}
    for key, value in arguments.items():
        try:
            rendered[key] = str(value).format(**context)
        except KeyError as e:
            return f"错误：工具链模板变量 {e} 未找到"
    return rendered


class ToolChain:
    """工具链 - 支持多个工具的顺序执行。"""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.steps: list[ChainStep] = []

    def add_step(
        self,
        tool_name: str,
        arguments: dict[str, str],
        output_key: Optional[str] = None,
    ) -> "ToolChain":
        step_index = len(self.steps)
        self.steps.append(
            ChainStep(
                tool_name=tool_name,
                arguments=arguments,
                output_key=output_key or f"step_{step_index}_result",
            )
        )
        return self

    def execute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        if not self.steps:
            return "错误：工具链没有任何步骤"

        ctx: dict[str, Any] = dict(context or {})
        ctx["input"] = initial_input

        print(f"🔗 开始执行工具链: {self.name}")

        for i, step in enumerate(self.steps, 1):
            rendered = _render_arguments(step.arguments, ctx)
            if isinstance(rendered, str):
                return rendered

            preview = next(iter(rendered.values()), "")
            print(f"  步骤 {i}: 使用 {step.tool_name} 处理 '{str(preview)[:50]}...'")

            result = registry.execute(step.tool_name, rendered)
            ctx[step.output_key] = result

            print(f"  ✅ 步骤 {i} 完成，结果长度: {len(result)} 字符")

        final_key = self.steps[-1].output_key
        print(f"🎉 工具链 '{self.name}' 执行完成")
        return str(ctx.get(final_key, ""))

    async def aexecute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        if not self.steps:
            return "错误：工具链没有任何步骤"

        ctx: dict[str, Any] = dict(context or {})
        ctx["input"] = initial_input

        print(f"🔗 开始执行工具链: {self.name}")

        for i, step in enumerate(self.steps, 1):
            rendered = _render_arguments(step.arguments, ctx)
            if isinstance(rendered, str):
                return rendered

            preview = next(iter(rendered.values()), "")
            print(f"  步骤 {i}: 使用 {step.tool_name} 处理 '{str(preview)[:50]}...'")

            result = await registry.aexecute(step.tool_name, rendered)
            ctx[step.output_key] = result

            print(f"  ✅ 步骤 {i} 完成，结果长度: {len(result)} 字符")

        final_key = self.steps[-1].output_key
        print(f"🎉 工具链 '{self.name}' 执行完成")
        return str(ctx.get(final_key, ""))


class ToolChainManager:
    """工具链管理器。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.chains: dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain) -> None:
        self.chains[chain.name] = chain
        print(f"✅ 工具链 '{chain.name}' 已注册")

    def execute_chain(
        self,
        chain_name: str,
        input_data: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        chain = self.chains.get(chain_name)
        if chain is None:
            return f"错误：工具链 '{chain_name}' 不存在"
        return chain.execute(self.registry, input_data, context)

    async def aexecute_chain(
        self,
        chain_name: str,
        input_data: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        chain = self.chains.get(chain_name)
        if chain is None:
            return f"错误：工具链 '{chain_name}' 不存在"
        return await chain.aexecute(self.registry, input_data, context)

    def list_chains(self) -> list[str]:
        return list(self.chains.keys())
