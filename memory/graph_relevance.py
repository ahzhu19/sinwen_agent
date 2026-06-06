"""语义图检索打分与 RRF 融合。"""

from __future__ import annotations


def reciprocal_rank_fusion(
    rank_lists: dict[str, dict[str, int]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """多路排序 RRF 融合。rank_lists: {track_name: {memory_id: rank}}，rank 从 1 开始。"""
    scores: dict[str, float] = {}
    for ranks in rank_lists.values():
        for memory_id, rank in ranks.items():
            scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)
    return scores


def build_ranks(scores: dict[str, float]) -> dict[str, int]:
    """按分数降序生成 rank（1-based）。"""
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {memory_id: index + 1 for index, (memory_id, _) in enumerate(ordered)}
