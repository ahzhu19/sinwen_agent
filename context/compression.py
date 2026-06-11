"""Compress 阶段：超 max_tokens 时的兜底压缩。

策略（仅在 enable_compression=True 时生效）：
1. 固定分区（Role/Task/State/Output）不压缩
2. Evidence 与 Context 各分一半 selectable 预算
3. 装不下的 packet 先尝试尾部截断，仍超出则丢弃
"""

from __future__ import annotations

from .config import ContextConfig
from .models import ContextPacket, SECTION_CONTEXT, SECTION_EVIDENCE
from .selector import SelectionResult
from .tokens import estimate_tokens


def compress_selection(
    selection: SelectionResult,
    *,
    config: ContextConfig,
    reserved_tokens: int,
) -> SelectionResult:
    """检查总 token 是否超限，必要时压缩 Evidence / Context。"""
    if not config.enable_compression:
        return selection

    selectable_budget = config.selectable_tokens
    total_used = reserved_tokens + selection.used_tokens
    if total_used <= config.max_tokens:
        return selection

    evidence = [
        packet
        for packet in selection.selected
        if packet.metadata.get("section") == SECTION_EVIDENCE
    ]
    context = [
        packet
        for packet in selection.selected
        if packet.metadata.get("section") == SECTION_CONTEXT
    ]

    # Evidence / Context 各分一半预算，避免一方独占
    compressed_evidence, evidence_dropped = _trim_packets(
        evidence,
        token_budget=max(0, selectable_budget // 2),
    )
    remaining_budget = max(
        0,
        selectable_budget - sum(packet.token_count for packet in compressed_evidence),
    )
    compressed_context, context_dropped = _trim_packets(context, token_budget=remaining_budget)

    compressed = compressed_evidence + compressed_context
    used_tokens = sum(packet.token_count for packet in compressed)
    dropped = list(selection.dropped) + evidence_dropped + context_dropped
    return SelectionResult(
        selected=compressed,
        dropped=dropped,
        used_tokens=used_tokens,
    )


def _trim_packets(
    packets: list[ContextPacket],
    *,
    token_budget: int,
) -> tuple[list[ContextPacket], list[ContextPacket]]:
    """按顺序装入预算；单条超出时尝试截断尾部。"""
    if token_budget <= 0:
        return [], list(packets)

    kept: list[ContextPacket] = []
    dropped: list[ContextPacket] = []
    used = 0
    for packet in packets:
        if used + packet.token_count <= token_budget:
            kept.append(packet)
            used += packet.token_count
            continue

        remaining = token_budget - used
        if remaining <= 0:
            dropped.append(packet)
            continue

        truncated = _truncate_packet(packet, max_tokens=remaining)
        if truncated is not None:
            kept.append(truncated)
            used += truncated.token_count
        dropped.append(packet)
    return kept, dropped


def _truncate_packet(packet: ContextPacket, *, max_tokens: int) -> ContextPacket | None:
    """从尾部逐字符截断，直至 token 数满足预算。"""
    if max_tokens <= 0:
        return None

    truncated_content = packet.content
    while truncated_content and estimate_tokens(truncated_content) > max_tokens:
        truncated_content = truncated_content[:-1]

    truncated_content = truncated_content.rstrip()
    if not truncated_content:
        return None

    if truncated_content != packet.content:
        truncated_content += "…"

    return ContextPacket(
        content=truncated_content,
        timestamp=packet.timestamp,
        token_count=estimate_tokens(truncated_content),
        relevance_score=packet.relevance_score,
        metadata={**packet.metadata, "truncated": True},
    )
