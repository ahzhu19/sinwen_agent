"""Structure 阶段：六分区固定骨架渲染。

分区结构（顺序固定，便于调试与 A/B 测试）：
  [Role & Policies]  系统指令与行为准则（预留，不参与竞争）
  [Task]             当前用户问题（预留）
  [State]            Agent 状态（预留）
  [Evidence]         记忆 + RAG 检索结果（竞争预算）
  [Context]          对话历史（竞争预算）
  [Output]           输出格式要求（预留）
"""

from __future__ import annotations

from .models import (
    ContextPacket,
    SECTION_CONTEXT,
    SECTION_EVIDENCE,
)


def render_context_text(
    *,
    system_instructions: str | None,
    user_query: str,
    state: str | None,
    output_requirements: str | None,
    evidence_packets: list[ContextPacket],
    context_packets: list[ContextPacket],
    default_output_requirements: str,
) -> str:
    """将各分区内容渲染为最终上下文字符串。"""
    evidence = _format_section_packets(evidence_packets, SECTION_EVIDENCE, "（无外部证据）")
    context = _format_section_packets(context_packets, SECTION_CONTEXT, "（无历史对话）")

    sections = [
        "[Role & Policies]",
        system_instructions.strip() if system_instructions else "（未指定）",
        "",
        "[Task]",
        user_query.strip(),
        "",
        "[State]",
        state.strip() if state else "（无）",
        "",
        "[Evidence]",
        evidence,
        "",
        "[Context]",
        context,
        "",
        "[Output]",
        output_requirements.strip()
        if output_requirements
        else default_output_requirements,
    ]
    return "\n".join(sections).strip()


def _format_section_packets(
    packets: list[ContextPacket],
    section: str,
    empty_label: str,
) -> str:
    """将同一分区的 packet 格式化为无序列表。"""
    matched = [packet for packet in packets if packet.metadata.get("section") == section]
    if not matched:
        return empty_label
    return "\n".join(f"- {packet.content}" for packet in matched)
