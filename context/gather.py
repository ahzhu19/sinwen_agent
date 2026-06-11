"""Gather 阶段：从各上下文源采集候选 ContextPacket。

每个 packet 通过 metadata["section"] 标记目标分区：
- SECTION_EVIDENCE：记忆检索 + RAG 片段
- SECTION_CONTEXT：对话历史
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING

from core.message import Message

from .config import ContextConfig
from .models import ContextPacket, SECTION_CONTEXT, SECTION_EVIDENCE
from .scoring import keyword_overlap_score
from .tokens import estimate_tokens

if TYPE_CHECKING:
    from memory.protocols import MemoryServiceProtocol
    from tools.builtin.rag_tool import RagManagerProtocol


def _record_content(record: Any) -> str:
    """兼容 MemoryRecord 与测试用 dict。"""
    if hasattr(record, "content"):
        return str(record.content)
    if isinstance(record, dict):
        return str(record.get("content", record))
    return str(record)


def _record_relevance(record: Any, *, query: str) -> float:
    """记忆相关性 = importance 与 query 关键词重叠的加权组合。

    MemoryService.search 不返回检索分数，因此用 importance 作代理，
    再叠加 keyword_overlap 以反映与当前 query 的匹配度。
    """
    if hasattr(record, "importance"):
        importance = float(record.importance)
        overlap = keyword_overlap_score(query, _record_content(record))
        return max(0.0, min(1.0, importance * 0.6 + overlap * 0.4))
    if isinstance(record, dict):
        importance = float(record.get("importance", 0.5))
        overlap = keyword_overlap_score(query, _record_content(record))
        return max(0.0, min(1.0, importance * 0.6 + overlap * 0.4))
    return keyword_overlap_score(query, _record_content(record))


def _record_timestamp(record: Any, *, fallback: datetime) -> datetime:
    """从 metadata 提取时间戳，供新近性评分使用。"""
    metadata: dict[str, Any] | None = None
    if hasattr(record, "metadata"):
        metadata = record.metadata
    elif isinstance(record, dict):
        metadata = record.get("metadata")
    if isinstance(metadata, dict):
        created = metadata.get("created_at") or metadata.get("occurred_at")
        if isinstance(created, (int, float)):
            return datetime.fromtimestamp(created)
        if isinstance(created, datetime):
            return created
    return fallback


def gather_history_packets(
    conversation_history: list[Message] | None,
    *,
    user_query: str,
) -> list[ContextPacket]:
    """将对话历史转为 [Context] 分区候选 packet。"""
    if not conversation_history:
        return []

    packets: list[ContextPacket] = []
    for message in conversation_history:
        content = f"[{message.role}] {message.content}"
        packets.append(
            ContextPacket(
                content=content,
                timestamp=message.timestamp,
                token_count=estimate_tokens(content),
                relevance_score=keyword_overlap_score(user_query, message.content),
                metadata={
                    "section": SECTION_CONTEXT,
                    "source": "history",
                    "role": message.role,
                },
            )
        )
    return packets


def gather_memory_packets(
    memory_service: MemoryServiceProtocol | None,
    *,
    user_query: str,
    config: ContextConfig,
    enabled_memory_types: list[str] | None,
    session_id: str | None,
    now: datetime,
) -> list[ContextPacket]:
    """从 MemoryService 检索记忆，转为 [Evidence] 分区候选 packet。

    检索类型取 config.memory_search_types 与 Tool 启用类型的交集，
    避免检索未启用的记忆模块。
    """
    if memory_service is None:
        return []

    enabled = set(enabled_memory_types or memory_service.memory_types)
    search_types = [
        memory_type
        for memory_type in config.memory_search_types
        if memory_type in enabled
    ]

    packets: list[ContextPacket] = []
    for memory_type in search_types:
        records = memory_service.search(
            user_query,
            memory_type,
            limit=config.memory_limit_per_type,
            session_id=session_id,
        )
        for record in records:
            content = _record_content(record)
            packets.append(
                ContextPacket(
                    content=content,
                    timestamp=_record_timestamp(record, fallback=now),
                    token_count=estimate_tokens(content),
                    relevance_score=_record_relevance(record, query=user_query),
                    metadata={
                        "section": SECTION_EVIDENCE,
                        "source": "memory",
                        "memory_type": memory_type,
                    },
                )
            )
    return packets


def gather_rag_packets(
    rag_manager: RagManagerProtocol | None,
    *,
    user_query: str,
    config: ContextConfig,
    now: datetime,
) -> list[ContextPacket]:
    """从 RagManager 检索知识片段，转为 [Evidence] 分区候选 packet。

    relevance_score 直接取自 RagSearchResult.score（向量检索分数）。
    """
    if rag_manager is None or not config.rag_enabled:
        return []

    results = rag_manager.search(
        query=user_query,
        top_k=config.rag_top_k,
        strategy="direct",
    )
    packets: list[ContextPacket] = []
    for index, result in enumerate(results, start=1):
        heading = " / ".join(result.chunk.heading_path) or "(无标题)"
        title = result.document.title or result.document.source_uri
        content = f"[{title} - {heading}] {result.chunk.content}"
        indexed_at = result.chunk.indexed_at or now
        packets.append(
            ContextPacket(
                content=content,
                timestamp=indexed_at,
                token_count=result.chunk.token_count or estimate_tokens(content),
                relevance_score=max(0.0, min(1.0, float(result.score))),
                metadata={
                    "section": SECTION_EVIDENCE,
                    "source": "rag",
                    "rank": index,
                    "document_id": result.document.id,
                    "chunk_id": result.chunk.id,
                },
            )
        )
    return packets
