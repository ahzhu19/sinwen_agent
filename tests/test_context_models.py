"""Context 模型与配置测试。"""

from datetime import datetime, timezone

import pytest

from context.config import ContextConfig
from context.models import ContextPacket


def test_context_packet_clamps_relevance_and_initializes_metadata() -> None:
    packet = ContextPacket(
        content="hello",
        timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc),
        token_count=3,
        relevance_score=1.5,
    )

    assert packet.relevance_score == 1.0
    assert packet.metadata == {}


def test_context_packet_preserves_metadata() -> None:
    packet = ContextPacket(
        content="evidence",
        timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc),
        token_count=5,
        relevance_score=-0.2,
        metadata={"section": "evidence"},
    )

    assert packet.relevance_score == 0.0
    assert packet.metadata == {"section": "evidence"}


def test_context_config_defaults_and_token_budget() -> None:
    config = ContextConfig()

    assert config.max_tokens == 8192
    assert config.reserve_ratio == 0.2
    assert config.min_relevance == 0.0
    assert config.enable_compression is True
    assert config.recency_weight == 0.5
    assert config.relevance_weight == 0.5
    assert config.reserved_tokens == 1638
    assert config.selectable_tokens == 6554


def test_context_config_clamps_ratios() -> None:
    config = ContextConfig(
        reserve_ratio=1.5,
        min_relevance=-0.1,
        recency_weight=2.0,
        relevance_weight=-1.0,
    )

    assert config.reserve_ratio == 1.0
    assert config.min_relevance == 0.0
    assert config.recency_weight == 1.0
    assert config.relevance_weight == 0.0


def test_context_config_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ContextConfig(max_tokens=0)
