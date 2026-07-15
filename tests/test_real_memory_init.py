"""A-04: 真实初始化路径测试 — Agent + MemoryTool 全链路。

覆盖 MemoryConfig → MemoryManager → MemoryService → MemoryTool → ToolRegistry → Agent，
不依赖 PG / Milvus / Neo4j，仅用 working memory 的 InMemoryStore。
"""

from __future__ import annotations

from typing import Any

from agents.react_agent import ReActAgent
from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse, ToolCall
from memory.config import MemoryConfig
from memory.service import MemoryService
from tools.builtin.memory_tool import MemoryTool


def _db_free_config() -> MemoryConfig:
    """无数据库、无向量 outbox 的 MemoryConfig。"""
    return MemoryConfig(
        database_url=None,
        enable_vector_outbox=False,
        enable_persistent_vector_outbox=False,
    )


def _real_service() -> MemoryService:
    """真实 MemoryService，仅 working memory，无需外部服务。"""
    return MemoryService(
        user_id="test_user",
        config=_db_free_config(),
        memory_types=["working"],
    )


# ---------------------------------------------------------------------------
# 1. MemoryTool → 真实 MemoryService → 真实 MemoryManager
# ---------------------------------------------------------------------------


def test_real_memory_tool_add_and_search() -> None:
    """通过真实栈添加记忆并检索。"""
    service = _real_service()
    tool = MemoryTool(
        user_id="test_user",
        session_id="sess-1",
        memory_service=service,
    )

    add_result = tool.execute(
        "add",
        content="Python 是一门解释型编程语言",
        memory_type="working",
        importance=0.8,
    )
    assert "✅" in add_result

    search_result = tool.execute(
        "search",
        query="Python 编程",
        memory_type="working",
        limit=5,
    )
    assert "Python" in search_result
    assert "解释型" in search_result


def test_real_memory_tool_stats() -> None:
    """stats 返回真实记忆条数。"""
    service = _real_service()
    tool = MemoryTool(
        user_id="test_user",
        session_id="sess-2",
        memory_service=service,
    )

    tool.execute("add", content="第一条记忆", memory_type="working", importance=0.5)
    tool.execute("add", content="第二条记忆", memory_type="working", importance=0.7)

    stats = tool.execute("stats")
    assert "working" in stats
    assert "2 条" in stats


def test_real_memory_tool_forget_dry_run() -> None:
    """dry_run 返回预览但不实际删除。"""
    service = _real_service()
    tool = MemoryTool(
        user_id="test_user",
        session_id="sess-3",
        memory_service=service,
    )

    tool.execute("add", content="低重要性记忆", memory_type="working", importance=0.1)
    tool.execute("add", content="高重要性记忆", memory_type="working", importance=0.9)

    dry = tool.execute(
        "forget",
        memory_type="working",
        importance_threshold=0.5,
        dry_run=True,
    )
    assert "低重要性" in dry
    assert "高重要性" not in dry

    # 未实际删除
    stats = tool.execute("stats")
    assert "2 条" in stats


def test_real_memory_tool_forget_execute() -> None:
    """forget 实际删除低重要性记忆。"""
    service = _real_service()
    tool = MemoryTool(
        user_id="test_user",
        session_id="sess-4",
        memory_service=service,
    )

    tool.execute("add", content="低重要性记忆", memory_type="working", importance=0.1)
    tool.execute("add", content="高重要性记忆", memory_type="working", importance=0.9)

    result = tool.execute("forget", memory_type="working", importance_threshold=0.5)
    assert "已遗忘 1 条" in result

    search = tool.execute("search", query="记忆", memory_type="working", limit=5)
    assert "高重要性" in search
    assert "低重要性" not in search


# ---------------------------------------------------------------------------
# 2. SimpleAgent — with_agent_tools 真实初始化
# ---------------------------------------------------------------------------


class _ScriptedLLM(BaseLLM):
    """按脚本返回 tool_calls 或最终文本。"""

    def __init__(self, script: list[ToolCall | str]) -> None:
        self.model = "scripted"
        self.client = None  # type: ignore[assignment]
        self._script = list(script)
        self._index = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = messages, temperature, kwargs
        if self._index < len(self._script):
            item = self._script[self._index]
            self._index += 1
            return item if isinstance(item, str) else None
        return "完成"

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        _ = messages, tools, tool_choice, temperature, kwargs
        if self._index < len(self._script):
            item = self._script[self._index]
            self._index += 1
            if isinstance(item, ToolCall):
                return LLMToolResponse(content=None, tool_calls=[item])
            return LLMToolResponse(content=item, tool_calls=None)
        return LLMToolResponse(content="完成", tool_calls=None)


def test_simple_agent_real_memory_add_and_search() -> None:
    """SimpleAgent 通过真实初始化路径调用 memory 工具。"""
    service = _real_service()

    agent = SimpleAgent.with_agent_tools(
        name="测试助手",
        llm=_ScriptedLLM([
            ToolCall(
                id="c1",
                name="memory",
                arguments='{"action":"add","content":"地球是太阳系第三颗行星","memory_type":"working","importance":0.8}',
            ),
            ToolCall(
                id="c2",
                name="memory",
                arguments='{"action":"search","query":"地球 太阳系","memory_type":"working","limit":5}',
            ),
            "地球是太阳系第三颗行星。",
        ]),
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        memory_types=["working"],
        memory_service=service,
        enable_memory_hooks=False,
    )

    answer = agent.run("告诉我关于地球的信息")
    assert "地球" in answer

    # 验证记忆真正写入了真实 manager
    stats = service.manager.memory_stats()
    assert stats["counts"]["working"] >= 1


# ---------------------------------------------------------------------------
# 3. ReActAgent — with_agent_tools 真实初始化
# ---------------------------------------------------------------------------


class _ReActLLM(BaseLLM):
    """按脚本返回 ReAct 格式文本。"""

    def __init__(self, responses: list[str]) -> None:
        self.model = "react-scripted"
        self.client = None  # type: ignore[assignment]
        self._responses = list(responses)
        self._index = 0

    def invoke(
        self,
        messages: LLMMessages,
        temperature: float = 0,
        **kwargs: Any,
    ) -> str | None:
        _ = messages, temperature, kwargs
        if self._index < len(self._responses):
            text = self._responses[self._index]
            self._index += 1
            return text
        return "Thought: 完成\nAction: Finish\nAction Input: 完成"

    def invoke_with_tools(
        self,
        messages: LLMMessages,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        temperature: float = 0,
        **kwargs: Any,
    ) -> LLMToolResponse:
        _ = messages, tools, tool_choice, temperature, kwargs
        return LLMToolResponse(content="完成", tool_calls=None)


def test_react_agent_real_memory_add_and_search() -> None:
    """ReActAgent 通过真实初始化路径调用 memory 工具。"""
    service = _real_service()

    agent = ReActAgent.with_agent_tools(
        name="React测试助手",
        llm=_ReActLLM([
            'Thought: 我需要先存储一条记忆\n'
            'Action: memory\n'
            'Action Input: {"action":"add","content":"水在零度以下会结冰","memory_type":"working","importance":0.7}',

            'Thought: 记忆已存储，现在给出回答\n'
            'Action: Finish\n'
            'Action Input: 水在零度以下会结冰。',
        ]),
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        memory_types=["working"],
        memory_service=service,
        enable_memory_hooks=False,
        enable_context=False,
    )

    answer = agent.run("水在什么温度结冰？")
    assert "水" in answer

    # 验证记忆真正写入了真实 manager
    stats = service.manager.memory_stats()
    assert stats["counts"]["working"] >= 1
