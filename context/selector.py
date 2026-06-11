"""Select 阶段：按综合分数过滤、排序、贪心装箱。

流程：
1. 丢弃 relevance_score < min_relevance 的 packet
2. 按 combined_score 降序排列
3. 从高到低装入 token_budget，装不下的进入 dropped
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .config import ContextConfig
from .models import ContextPacket
from .scoring import combined_score


@dataclass
class SelectionResult:
    """选择结果。"""

    selected: list[ContextPacket]
    dropped: list[ContextPacket]
    used_tokens: int


def select_packets(
    packets: list[ContextPacket],
    *,
    config: ContextConfig,
    token_budget: int,
    now: datetime,
) -> SelectionResult:
    """在 token 预算内选取高分 packet。

    Args:
        packets: Gather 阶段产出的全部候选。
        config: 上下文配置（含 min_relevance 与权重）。
        token_budget: 可用于 Evidence + Context 的 token 上限。
        now: 新近性评分基准时间。

    Returns:
        选中列表、丢弃列表及实际占用 token 数。
    """
    if not packets:
        return SelectionResult(selected=[], dropped=[], used_tokens=0)

    below_threshold = [
        packet
        for packet in packets
        if packet.relevance_score < config.min_relevance
    ]
    eligible = [
        packet
        for packet in packets
        if packet.relevance_score >= config.min_relevance
    ]

    scored = [
        (combined_score(packet, config=config, now=now), packet) for packet in eligible
    ]
    scored.sort(key=lambda item: item[0], reverse=True)

    if token_budget <= 0:
        return SelectionResult(
            selected=[],
            dropped=below_threshold + eligible,
            used_tokens=0,
        )

    selected: list[ContextPacket] = []
    dropped: list[ContextPacket] = list(below_threshold)
    used_tokens = 0

    for score, packet in scored:
        if used_tokens + packet.token_count <= token_budget:
            packet.metadata["combined_score"] = score
            selected.append(packet)
            used_tokens += packet.token_count
        else:
            packet.metadata["combined_score"] = score
            dropped.append(packet)

    return SelectionResult(selected=selected, dropped=dropped, used_tokens=used_tokens)
