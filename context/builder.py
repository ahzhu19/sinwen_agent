"""ContextBuilder：统一上下文构建入口。

流水线：Gather → Select → Structure → Compress
- Gather：从对话历史、记忆、RAG 采集候选 ContextPacket
- Select：按「相关性 + 新近性」评分，在 token 预算内贪心选取
- Structure：渲染六分区固定骨架（Role / Task / State / Evidence / Context / Output）
- Compress：超预算时截断或丢弃低价值 packet
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from core.message import Message

from .compression import compress_selection
from .config import ContextConfig
from .gather import gather_history_packets, gather_memory_packets, gather_rag_packets
from .models import BuiltContext, SECTION_CONTEXT, SECTION_EVIDENCE
from .selector import select_packets
from .template import render_context_text
from .tokens import estimate_tokens

if TYPE_CHECKING:
    from tools.builtin.memory_tool import MemoryTool
    from tools.builtin.rag_tool import RagTool


class ContextBuilder:
    """统一上下文构建入口。

    构造时接受 MemoryTool / RagTool，但 Gather 阶段穿透到
    ``memory_service`` / ``rag_manager`` 获取结构化数据，
    而非调用 ``tool.run()`` 解析格式化字符串。
    """

    def __init__(
        self,
        *,
        config: ContextConfig | None = None,
        memory_tool: MemoryTool | None = None,
        rag_tool: RagTool | None = None,
    ) -> None:
        self.config = config or ContextConfig()
        # 穿透 Tool 层，直接访问底层 Service / Manager
        self._memory_service = memory_tool.memory_service if memory_tool else None
        self._enabled_memory_types = (
            list(memory_tool.memory_types) if memory_tool is not None else None
        )
        self._default_session_id = (
            memory_tool.current_session_id if memory_tool is not None else None
        )
        self._rag_manager = rag_tool.rag_manager if rag_tool else None

    def build(
        self,
        *,
        user_query: str,
        conversation_history: list[Message] | None = None,
        system_instructions: str | None = None,
        state: str | None = None,
        output_requirements: str | None = None,
        session_id: str | None = None,
        now: datetime | None = None,
    ) -> BuiltContext:
        """构建完整上下文。

        Args:
            user_query: 当前用户问题，写入 [Task] 分区。
            conversation_history: 对话历史，候选进入 [Context] 分区。
            system_instructions: 系统指令，写入 [Role & Policies] 分区。
            state: Agent 当前状态，写入 [State] 分区。
            output_requirements: 输出格式要求，写入 [Output] 分区。
            session_id: 记忆检索会话 ID，默认取 memory_tool 的 current_session_id。
            now: 评分基准时间，默认当前时间；测试时可注入固定值。

        Returns:
            BuiltContext，含六分区文本、LLM 消息列表与调试统计。
        """
        current_time = now or datetime.now()
        output_text = output_requirements or self.config.default_output_requirements

        # 固定分区（Role/Task/State/Output）先估算 token，剩余预算给 Evidence/Context
        reserved_tokens = self._estimate_reserved_tokens(
            system_instructions=system_instructions,
            user_query=user_query,
            state=state,
            output_requirements=output_text,
        )
        selectable_budget = max(
            0,
            min(self.config.selectable_tokens, self.config.max_tokens - reserved_tokens),
        )

        # --- Gather ---
        packets = (
            gather_history_packets(conversation_history, user_query=user_query)
            + gather_memory_packets(
                self._memory_service,
                user_query=user_query,
                config=self.config,
                enabled_memory_types=self._enabled_memory_types,
                session_id=session_id or self._default_session_id,
                now=current_time,
            )
            + gather_rag_packets(
                self._rag_manager,
                user_query=user_query,
                config=self.config,
                now=current_time,
            )
        )

        # --- Select ---
        selection = select_packets(
            packets,
            config=self.config,
            token_budget=selectable_budget,
            now=current_time,
        )

        # --- Compress（仅在 enable_compression 且超 max_tokens 时生效）---
        selection = compress_selection(
            selection,
            config=self.config,
            reserved_tokens=reserved_tokens,
        )

        evidence_packets = [
            packet
            for packet in selection.selected
            if packet.metadata.get("section") == SECTION_EVIDENCE
        ]
        context_packets = [
            packet
            for packet in selection.selected
            if packet.metadata.get("section") == SECTION_CONTEXT
        ]

        # --- Structure ---
        text = render_context_text(
            system_instructions=system_instructions,
            user_query=user_query,
            state=state,
            output_requirements=output_requirements,
            evidence_packets=evidence_packets,
            context_packets=context_packets,
            default_output_requirements=self.config.default_output_requirements,
        )

        stats = self._build_stats(
            text=text,
            reserved_tokens=reserved_tokens,
            selectable_budget=selectable_budget,
            selection=selection,
            evidence_packets=evidence_packets,
            context_packets=context_packets,
        )
        # 默认单条 system 消息，Agent 可直接传给 LLM
        messages = [{"role": "system", "content": text}]
        return BuiltContext(text=text, messages=messages, stats=stats)

    def _estimate_reserved_tokens(
        self,
        *,
        system_instructions: str | None,
        user_query: str,
        state: str | None,
        output_requirements: str,
    ) -> int:
        """渲染空 Evidence/Context 的骨架，估算固定分区实际 token 占用。"""
        skeleton = render_context_text(
            system_instructions=system_instructions,
            user_query=user_query,
            state=state,
            output_requirements=output_requirements,
            evidence_packets=[],
            context_packets=[],
            default_output_requirements=self.config.default_output_requirements,
        )
        return estimate_tokens(skeleton)

    def _build_stats(
        self,
        *,
        text: str,
        reserved_tokens: int,
        selectable_budget: int,
        selection: Any,
        evidence_packets: list[Any],
        context_packets: list[Any],
    ) -> dict[str, Any]:
        """组装调试统计，便于 A/B 测试与预算排查。"""
        return {
            "max_tokens": self.config.max_tokens,
            "reserved_tokens": reserved_tokens,
            "selectable_budget": selectable_budget,
            "selected_packets": len(selection.selected),
            "dropped_packets": len(selection.dropped),
            "evidence_packets": len(evidence_packets),
            "context_packets": len(context_packets),
            "selectable_used_tokens": selection.used_tokens,
            "total_tokens": estimate_tokens(text),
        }
