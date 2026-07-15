"""PlanAndSolveAgent memory integration tests (AG-02)."""

from __future__ import annotations

from typing import Any

from agents.plan_and_solve_agent import PlanAndSolveAgent
from core.llm import BaseLLM, LLMMessages
from core.llm_types import LLMToolResponse


class ScriptedLLM(BaseLLM):
    def __init__(self, responses: list[str | None]) -> None:
        self.model = "fake"
        self.client = None  # type: ignore[assignment]
        self.responses = responses
        self.calls = 0
        self.all_messages: list[LLMMessages] = []

    def invoke(self, messages: LLMMessages, temperature: float = 0, **kwargs: Any) -> str | None:
        self.all_messages.append(messages)
        r = self.responses[self.calls]
        self.calls += 1
        return r

    def invoke_with_tools(self, *args: Any, **kwargs: Any) -> LLMToolResponse:
        return LLMToolResponse(content=None, tool_calls=None)


class CapturingMemoryService:
    """Fake MemoryServiceProtocol that records all calls."""

    def __init__(self, search_results: list[Any] | None = None) -> None:
        self.added: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self._search_results = search_results or []

    def add(self, content: str, memory_type: str, importance: float, metadata: dict[str, Any]) -> str:
        payload = {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "metadata": metadata,
        }
        self.added.append(payload)
        return "mem-fake-id"

    def search(self, query: str, memory_type: str, limit: int = 5, session_id: str | None = None) -> list[Any]:
        self.searches.append({
            "query": query,
            "memory_type": memory_type,
            "limit": limit,
            "session_id": session_id,
        })
        return self._search_results

    def retrieve_context(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def record_interaction(self, *args: Any, **kwargs: Any) -> str:
        return "mem-fake-id"


def test_plan_solve_retrieves_past_plans_before_planning() -> None:
    """AG-02: 规划前从 episodic 检索相似历史计划。"""
    llm = ScriptedLLM([
        '["第一步", "第二步"]',
        "步骤1结果",
        "步骤2结果",
        "最终答案",
    ])
    service = CapturingMemoryService(
        search_results=[{"id": "ep-1", "content": "之前做过类似问题"}]
    )
    agent = PlanAndSolveAgent(
        name="plan-solve",
        llm=llm,
        memory_service=service,
        enable_memory=True,
    )

    agent.run("1+2等于多少")

    # Should have searched episodic memory
    assert len(service.searches) == 1
    search = service.searches[0]
    assert search["memory_type"] == "episodic"
    assert search["query"] == "1+2等于多少"
    assert search["limit"] == 3

    # Planner should have received the episodic context
    planner_messages = llm.all_messages[0]
    assert len(planner_messages) == 3  # system + user + episodic context
    assert "历史相似计划参考" in planner_messages[2]["content"]
    assert "之前做过类似问题" in planner_messages[2]["content"]


def test_plan_solve_stores_plan_to_episodic_after_completion() -> None:
    """AG-02: 完成后将计划+结果存入 episodic 记忆。"""
    llm = ScriptedLLM([
        '["分析", "计算"]',
        "分析结果",
        "计算结果",
        "最终汇总",
    ])
    service = CapturingMemoryService()
    agent = PlanAndSolveAgent(
        name="plan-solve",
        llm=llm,
        memory_service=service,
        enable_memory=True,
    )

    agent.run("数学问题")

    assert len(service.added) == 1
    record = service.added[0]
    assert record["memory_type"] == "episodic"
    assert record["importance"] == 0.6
    assert record["metadata"]["agent_type"] == "plan_and_solve"
    assert record["metadata"]["plan_steps"] == 2
    assert "数学问题" in record["content"]
    assert "最终汇总" in record["content"]


def test_plan_solve_no_memory_when_disabled() -> None:
    """enable_memory=False 时不检索也不存储。"""
    llm = ScriptedLLM([
        '["步骤"]',
        "结果",
        "最终答案",
    ])
    service = CapturingMemoryService()
    agent = PlanAndSolveAgent(
        name="plan-solve",
        llm=llm,
        memory_service=service,
        enable_memory=False,
    )

    agent.run("问题")

    assert len(service.searches) == 0
    assert len(service.added) == 0


def test_plan_solve_no_memory_when_service_is_none() -> None:
    """memory_service=None 时不检索也不存储。"""
    llm = ScriptedLLM([
        '["步骤"]',
        "结果",
        "最终答案",
    ])
    agent = PlanAndSolveAgent(
        name="plan-solve",
        llm=llm,
        memory_service=None,
        enable_memory=True,
    )

    result = agent.run("问题")

    assert result == "最终答案"


def test_plan_solve_search_failure_does_not_crash() -> None:
    """检索失败不应影响 Agent 正常运行。"""

    class FailingSearchService(CapturingMemoryService):
        def search(self, *args: Any, **kwargs: Any) -> list[Any]:
            raise RuntimeError("DB down")

    llm = ScriptedLLM([
        '["步骤"]',
        "结果",
        "最终答案",
    ])
    service = FailingSearchService()
    agent = PlanAndSolveAgent(
        name="plan-solve",
        llm=llm,
        memory_service=service,
        enable_memory=True,
    )

    result = agent.run("问题")

    assert result == "最终答案"
    # Even if search failed, should still try to record
    assert len(service.added) == 1


def test_plan_solve_write_failure_does_not_crash() -> None:
    """写入失败不应影响 Agent 正常返回。"""

    class FailingWriteService(CapturingMemoryService):
        def add(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("DB down")

    llm = ScriptedLLM([
        '["步骤"]',
        "结果",
        "最终答案",
    ])
    service = FailingWriteService()
    agent = PlanAndSolveAgent(
        name="plan-solve",
        llm=llm,
        memory_service=service,
        enable_memory=True,
    )

    result = agent.run("问题")

    assert result == "最终答案"


def test_plan_solve_no_episodic_context_when_no_results() -> None:
    """检索无结果时不注入 episodic context。"""
    llm = ScriptedLLM([
        '["步骤"]',
        "结果",
        "最终答案",
    ])
    service = CapturingMemoryService(search_results=[])
    agent = PlanAndSolveAgent(
        name="plan-solve",
        llm=llm,
        memory_service=service,
        enable_memory=True,
    )

    agent.run("问题")

    # Planner should only have system + user, no episodic context
    planner_messages = llm.all_messages[0]
    assert len(planner_messages) == 2


def test_plan_solve_with_agent_tools_creates_service() -> None:
    """with_agent_tools(enable_memory=True) 自动创建 MemoryService。"""
    from memory.service import MemoryService

    llm = ScriptedLLM([
        '["步骤"]',
        "结果",
        "最终答案",
    ])
    agent = PlanAndSolveAgent.with_agent_tools(
        name="plan-solve",
        llm=llm,
        enable_memory=True,
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        memory_types=["working"],
    )

    assert isinstance(agent.memory_service, MemoryService)
    assert agent.enable_memory is True
