"""ContextBuilder 集成测试。"""

from datetime import datetime, timezone

from core.message import Message
from memory.modules.base import MemoryRecord
from memory.service import MemoryService
from rag.models import RagChunk, RagDocument, RagSearchResult
from tests.memory_fakes import FakeMemoryManager
from tests.test_rag_tool import FakeRagManager

from context import BuiltContext, ContextBuilder, ContextConfig
from tools.builtin.memory_tool import MemoryTool
from tools.builtin.rag_tool import RagTool


class SearchableFakeMemoryManager(FakeMemoryManager):
    def search_memory(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
        **kwargs,
    ) -> list[MemoryRecord]:
        super().search_memory(query, memory_type, limit, session_id, **kwargs)
        return [
            MemoryRecord(
                id="mem-semantic-1",
                content="用户正在开发数据分析工具,使用Python和Pandas",
                memory_type="semantic",
                importance=0.8,
                metadata={"created_at": datetime(2026, 6, 12, tzinfo=timezone.utc).timestamp()},
            ),
            MemoryRecord(
                id="mem-episodic-1",
                content="已完成CSV读取模块的开发",
                memory_type="episodic",
                importance=0.7,
                metadata={"created_at": datetime(2026, 6, 12, 1, tzinfo=timezone.utc).timestamp()},
            ),
        ]


def _conversation_history() -> list[Message]:
    base = datetime(2026, 6, 12, 2, tzinfo=timezone.utc)
    return [
        Message(content="我正在开发一个数据分析工具", role="user", timestamp=base),
        Message(
            content="很好!数据分析工具通常需要处理大量数据。",
            role="assistant",
            timestamp=base,
        ),
        Message(
            content="我打算使用Python和Pandas,已经完成了CSV读取模块",
            role="user",
            timestamp=base,
        ),
    ]


def test_context_builder_returns_built_context_with_sections() -> None:
    memory_service = MemoryService(
        user_id="user123",
        memory_types=["working", "episodic", "semantic"],
        manager=SearchableFakeMemoryManager(),
    )
    memory_tool = MemoryTool(
        user_id="user123",
        memory_types=["working", "episodic", "semantic"],
        memory_service=memory_service,
    )
    rag_tool = RagTool(rag_manager=FakeRagManager())
    config = ContextConfig(max_tokens=3000, reserve_ratio=0.2, min_relevance=0.0)

    builder = ContextBuilder(memory_tool=memory_tool, rag_tool=rag_tool, config=config)
    result = builder.build(
        user_query="如何优化Pandas的内存占用?",
        conversation_history=_conversation_history(),
        system_instructions="你是一位资深的Python数据工程顾问。",
    )

    assert isinstance(result, BuiltContext)
    assert "[Role & Policies]" in result.text
    assert "[Task]" in result.text
    assert "[Evidence]" in result.text
    assert "[Context]" in result.text
    assert "Pandas" in result.text
    assert "Milvus setup" in result.text
    assert result.messages == [{"role": "system", "content": result.text}]
    assert result.stats["selected_packets"] >= 1
    assert result.stats["total_tokens"] > 0


def test_context_builder_uses_memory_service_not_tool_run() -> None:
    manager = SearchableFakeMemoryManager()
    memory_service = MemoryService(
        user_id="user123",
        memory_types=["semantic"],
        manager=manager,
    )
    memory_tool = MemoryTool(
        user_id="user123",
        memory_types=["semantic"],
        memory_service=memory_service,
    )

    builder = ContextBuilder(memory_tool=memory_tool, config=ContextConfig(min_relevance=0.0))
    builder.build(user_query="Pandas 内存")

    assert manager.searches
    assert manager.searches[0]["query"] == "Pandas 内存"


def test_context_builder_respects_min_relevance() -> None:
    manager = SearchableFakeMemoryManager()
    memory_service = MemoryService(
        user_id="user123",
        memory_types=["semantic"],
        manager=manager,
    )
    memory_tool = MemoryTool(
        user_id="user123",
        memory_types=["semantic"],
        memory_service=memory_service,
    )
    config = ContextConfig(max_tokens=3000, min_relevance=0.99)

    result = ContextBuilder(memory_tool=memory_tool, config=config).build(
        user_query="无关话题",
        conversation_history=[
            Message(content="完全无关的内容", role="user", timestamp=datetime.now()),
        ],
    )

    assert result.stats["dropped_packets"] >= 1
    assert "（无历史对话）" in result.text or result.stats["context_packets"] == 0
