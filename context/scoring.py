"""Select 阶段评分：相关性 + 新近性。

设计原则：不引入来源/优先级维度，记忆、RAG、历史统一评分竞争预算。
综合分 = relevance_weight × relevance_score + recency_weight × recency_score
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .config import ContextConfig
from .models import ContextPacket
from .tokens import tokenize


def keyword_overlap_score(query: str, text: str) -> float:
    """简单关键词重叠率，用于历史消息与无检索分数的记忆记录。

    返回 query 中有多少比例的 token 出现在 text 中（0.0-1.0）。
    """
    query_tokens = {token.lower() for token in tokenize(query) if token.strip()}
    if not query_tokens:
        return 0.5
    text_tokens = {token.lower() for token in tokenize(text) if token.strip()}
    overlap = len(query_tokens & text_tokens)
    return min(1.0, overlap / len(query_tokens))


def _to_utc_timestamp(value: datetime) -> float:
    """统一 naive/aware datetime，避免相减时区错误。"""
    if value.tzinfo is None:
        return value.timestamp()
    return value.astimezone(timezone.utc).timestamp()


def recency_score(timestamp: datetime, *, now: datetime, half_life_seconds: float) -> float:
    """指数衰减新近性分数，半衰期由 config.recency_half_life_seconds 控制。

    公式：exp(-age × ln2 / half_life)，刚发生时 ≈ 1.0，经过一个半衰期后 ≈ 0.5。
    """
    if half_life_seconds <= 0:
        return 1.0
    age_seconds = max(0.0, _to_utc_timestamp(now) - _to_utc_timestamp(timestamp))
    return math.exp(-age_seconds * math.log(2) / half_life_seconds)


def combined_score(
    packet: ContextPacket,
    *,
    config: ContextConfig,
    now: datetime,
) -> float:
    """计算 packet 的最终排序分数。"""
    recency = recency_score(
        packet.timestamp,
        now=now,
        half_life_seconds=config.recency_half_life_seconds,
    )
    return (
        config.relevance_weight * packet.relevance_score
        + config.recency_weight * recency
    )
