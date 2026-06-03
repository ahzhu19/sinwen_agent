"""SimpleAgent + RagTool integration tests."""

from typing import Any

from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse, ToolCall
from tools.agent_registry import create_agent_tool_registry
from tools.builtin.rag_tool import RagTool


class FakeRagManager:
    def ingest(self, source: str, source_type: str = "file", metadata=None):
        from rag.models import RagDocument
        from datetime import datetime, timezone

        _ = source_type, metadata
        now = datetime.now(timezone.utc)
        return RagDocument(
            id="doc-12345678-abcd",
            source_uri=source,
            source_type="file",
            title="demo.md",
            mime_type="text/markdown",
            content_hash="hash",
            markdown="# Demo",
            status="indexed",
            metadata={},
            created_at=now,
            updated_at=now,
        )

    def search(self, query: str, top_k: int = 5, strategy: str = "direct"):
        return []

    def answer(self, query: str, top_k: int = 5, strategy: str = "direct"):
        from rag.models import RagAnswer

        _ = query, top_k, strategy
        return RagAnswer(answer="来自知识库的回答", sources=[])

    def list_documents(self, limit: int = 50):
        return []

    def delete(self, document_id: str) -> None:
        _ = document_id

    def reindex(self, document_id: str):
        return self.ingest("/tmp/x.md")

    def stats(self):
        return {
            "document_count": 1,
            "chunk_count": 2,
            "indexed_chunk_count": 2,
            "collection": "test",
        }


class FakeRagLLM(BaseLLM):
    def __init__(self) -> None:
        self.model = "fake-rag"
        self.client = None  # type: ignore[assignment]
        self._round = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = messages, temperature, kwargs
        return "最终回答"

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        _ = messages, tools, tool_choice, temperature, kwargs
        self._round += 1
        if self._round == 1:
            return LLMToolResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_rag",
                        name="rag",
                        arguments='{"action": "stats"}',
                    )
                ],
            )
        return LLMToolResponse(content="已根据 RAG 统计生成回答", tool_calls=None)


def test_create_agent_tool_registry_includes_rag() -> None:
    registry = create_agent_tool_registry(
        enable_search=False,
        enable_calculator=False,
        enable_rag=True,
    )
    assert "rag" in registry.list_tools()


def test_simple_agent_with_agent_tools_factory() -> None:
    agent = SimpleAgent.with_agent_tools(
        name="助手",
        llm=FakeRagLLM(),
        enable_search=False,
        enable_calculator=False,
        enable_rag=True,
    )
    assert "rag" in agent.list_tools()


def test_simple_agent_executes_rag_tool() -> None:
    registry = create_agent_tool_registry(
        enable_search=False,
        enable_calculator=False,
        enable_rag=True,
    )
    registry.unregister_tool("rag")
    registry.register_tool(RagTool(rag_manager=FakeRagManager()))

    agent = SimpleAgent(
        name="RAG 助手",
        llm=FakeRagLLM(),
        tool_registry=registry,
        enable_tool_calling=True,
    )
    answer = agent.run("知识库里有多少文档？")
    assert "RAG 统计" in answer or "已根据 RAG 统计" in answer
