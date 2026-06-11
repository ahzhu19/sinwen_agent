# Memory Service Tool Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote memory from a Tool-owned subsystem to an internal `MemoryService`, while keeping `MemoryTool` as a thin adapter with unchanged external behavior.

**Architecture:** Add `memory/service.py` as the stable application boundary over the existing `MemoryManager`. `MemoryTool` and tool registry construction will depend on the service interface, not directly on `MemoryManager`, so later Agent runtime hooks can call the same service without going through LLM tool calls.

**Tech Stack:** Python 3.11+, existing `MemoryManager`, `MemoryConfig`, `MemoryRecord`, pytest, current fake memory managers.

---

## File Structure

- Create `memory/service.py`: owns `MemoryService`, a thin façade over `MemoryManager` with runtime-oriented method names and default manager construction.
- Modify `memory/protocols.py`: add `MemoryServiceProtocol`; keep `MemoryManagerProtocol` temporarily for fake compatibility during the migration.
- Modify `tools/builtin/memory_tool.py`: replace direct manager construction with service construction and service calls.
- Modify `tools/agent_registry.py`: allow callers to inject a `MemoryServiceProtocol`; keep current defaults for backward compatibility.
- Create `tests/test_memory_service.py`: verifies service delegation and default manager construction boundaries.
- Modify `tests/test_memory_tool.py`: update fake injection to use a service fake or wrap the existing fake manager with `MemoryService`.
- Modify `tests/test_agents_with_memory.py`, `tests/test_simple_agent_memory.py`, `tests/test_react_agent_with_tools.py` only if direct `MemoryTool(memory_manager=...)` construction fails after the adapter change.
- Modify `docs/architecture/memory.md`: document `MemoryService` as the internal boundary and `MemoryTool` as the external adapter.
- Modify `docs/system-issues.md`: update A-02 / A-05 notes to reflect the first step toward service extraction.

---

### Task 1: Define MemoryService Protocol

**Files:**
- Modify: `memory/protocols.py`
- Test: no standalone test yet; Task 2 exercises the protocol through `MemoryService`.

- [ ] **Step 1: Add `MemoryServiceProtocol`**

Append this protocol after `MemoryManagerProtocol` in `memory/protocols.py`:

```python
class MemoryServiceProtocol(Protocol):
    def add(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        ...

    def search(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[Any]:
        ...

    def update(
        self,
        memory_id: str,
        memory_type: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        ...

    def remove(self, memory_id: str, memory_type: str) -> None:
        ...

    def forget(
        self,
        memory_type: str = "working",
        *,
        strategy: str = "importance",
        session_id: str | None = None,
        importance_threshold: float | None = None,
        older_than_days: int | None = None,
        limit: int | None = None,
    ) -> int:
        ...

    def consolidate(
        self,
        session_id: str,
        *,
        importance_threshold: float = 0.5,
    ) -> list[str]:
        ...

    def clear(
        self,
        memory_type: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, int]:
        ...

    def stats(self, session_id: str | None = None) -> dict[str, Any]:
        ...

    def summary(
        self,
        session_id: str | None = None,
        limit_per_type: int = 3,
    ) -> dict[str, list[Any]]:
        ...
```

- [ ] **Step 2: Run import check**

Run:

```bash
uv run python - <<'PY'
from memory.protocols import MemoryServiceProtocol
print(MemoryServiceProtocol)
PY
```

Expected: prints the protocol class without import errors.

---

### Task 2: Add MemoryService Façade

**Files:**
- Create: `memory/service.py`
- Create: `tests/test_memory_service.py`

- [ ] **Step 1: Write service delegation tests**

Create `tests/test_memory_service.py`:

```python
"""MemoryService tests."""

from __future__ import annotations

from memory.service import MemoryService
from tests.memory_fakes import FakeMemoryManager


def test_memory_service_delegates_crud_operations() -> None:
    manager = FakeMemoryManager(memory_id="memory_123")
    service = MemoryService(manager=manager)

    memory_id = service.add(
        content="记住用户喜欢 Python",
        memory_type="working",
        importance=0.8,
        metadata={"session_id": "session_1"},
    )
    results = service.search(
        query="Python",
        memory_type="working",
        limit=3,
        session_id="session_1",
    )
    updated_id = service.update(
        "memory_123",
        "working",
        content="用户喜欢 Python 和 Neo4j",
        importance=0.9,
        metadata={"topic": "preference"},
    )
    service.remove("memory_123", "working")

    assert memory_id == "memory_123"
    assert updated_id == "memory_123"
    assert results
    assert manager.added[0]["content"] == "记住用户喜欢 Python"
    assert manager.searches[0]["query"] == "Python"
    assert manager.updated[0]["content"] == "用户喜欢 Python 和 Neo4j"
    assert manager.removed == [("memory_123", "working")]


def test_memory_service_delegates_lifecycle_operations() -> None:
    manager = FakeMemoryManager()
    service = MemoryService(manager=manager)

    forgotten = service.forget(
        "working",
        strategy="importance_ttl",
        session_id="session_1",
        importance_threshold=0.2,
        older_than_days=7,
        limit=5,
    )
    consolidated = service.consolidate("session_1", importance_threshold=0.7)
    cleared = service.clear(memory_type="working", session_id="session_1")
    stats = service.stats(session_id="session_1")
    summary = service.summary(session_id="session_1", limit_per_type=2)

    assert forgotten == manager.forgotten_count
    assert consolidated == manager.consolidated_ids
    assert cleared == manager.cleared
    assert stats["user_id"] == manager.stats_user_id
    assert "working" in summary
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
uv run pytest tests/test_memory_service.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'memory.service'`.

- [ ] **Step 3: Implement `MemoryService`**

Create `memory/service.py`:

```python
"""Memory service: internal boundary over MemoryManager.

MemoryTool is only one adapter for this service. Agent runtime hooks should use
this class directly instead of simulating tool calls.
"""

from __future__ import annotations

from typing import Any

from .config import MemoryConfig
from .manager import MemoryManager
from .protocols import MemoryManagerProtocol


class MemoryService:
    """Application-facing memory API backed by the existing MemoryManager."""

    def __init__(
        self,
        *,
        user_id: str = "default_user",
        config: MemoryConfig | None = None,
        memory_types: list[str] | None = None,
        manager: MemoryManagerProtocol | None = None,
    ) -> None:
        self.user_id = user_id
        self.config = config or MemoryConfig.from_env()
        self.memory_types = list(memory_types or ["working"])
        self._manager = manager or MemoryManager(
            config=self.config,
            user_id=user_id,
            enable_working="working" in self.memory_types,
            enable_episodic="episodic" in self.memory_types,
            enable_semantic="semantic" in self.memory_types,
            enable_perceptual="perceptual" in self.memory_types,
        )

    @property
    def manager(self) -> MemoryManagerProtocol:
        return self._manager

    def add(
        self,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any],
    ) -> str:
        return self._manager.add_memory(content, memory_type, importance, metadata)

    def search(
        self,
        query: str,
        memory_type: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[Any]:
        return self._manager.search_memory(query, memory_type, limit, session_id)

    def update(
        self,
        memory_id: str,
        memory_type: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self._manager.update_memory(
            memory_id,
            memory_type,
            content=content,
            importance=importance,
            metadata=metadata,
        )

    def remove(self, memory_id: str, memory_type: str) -> None:
        self._manager.remove_memory(memory_id, memory_type)

    def forget(
        self,
        memory_type: str = "working",
        *,
        strategy: str = "importance",
        session_id: str | None = None,
        importance_threshold: float | None = None,
        older_than_days: int | None = None,
        limit: int | None = None,
    ) -> int:
        return self._manager.forget_memories(
            memory_type,
            strategy=strategy,
            session_id=session_id,
            importance_threshold=importance_threshold,
            older_than_days=older_than_days,
            limit=limit,
        )

    def consolidate(
        self,
        session_id: str,
        *,
        importance_threshold: float = 0.5,
    ) -> list[str]:
        return self._manager.consolidate_working_to_episodic(
            session_id,
            importance_threshold=importance_threshold,
        )

    def clear(
        self,
        memory_type: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, int]:
        return self._manager.clear_memories(memory_type=memory_type, session_id=session_id)

    def stats(self, session_id: str | None = None) -> dict[str, Any]:
        return self._manager.memory_stats(session_id=session_id)

    def summary(
        self,
        session_id: str | None = None,
        limit_per_type: int = 3,
    ) -> dict[str, list[Any]]:
        return self._manager.memory_summary(
            session_id=session_id,
            limit_per_type=limit_per_type,
        )
```

- [ ] **Step 4: Run service tests**

Run:

```bash
uv run pytest tests/test_memory_service.py -q
```

Expected: PASS.

---

### Task 3: Update MemoryTool to Use MemoryService

**Files:**
- Modify: `tools/builtin/memory_tool.py`
- Modify: `tests/test_memory_tool.py`

- [ ] **Step 1: Write or adjust adapter tests**

In `tests/test_memory_tool.py`, keep existing behavior tests and add one constructor test:

```python
from memory.service import MemoryService


def test_memory_tool_accepts_memory_service() -> None:
    manager = FakeMemoryManager("service_mem_123")
    service = MemoryService(manager=manager)
    tool = MemoryTool(user_id="user123", memory_service=service)

    result = tool.execute(
        "add",
        content="用户喜欢 Python",
        memory_type="working",
        importance=0.8,
    )

    assert "service_mem" in result
    assert manager.added[0]["content"] == "用户喜欢 Python"
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```bash
uv run pytest tests/test_memory_tool.py::test_memory_tool_accepts_memory_service -q
```

Expected: FAIL with `TypeError: MemoryTool.__init__() got an unexpected keyword argument 'memory_service'`.

- [ ] **Step 3: Change `MemoryTool` constructor**

In `tools/builtin/memory_tool.py`, replace manager imports and constructor wiring:

```python
from memory.protocols import MemoryServiceProtocol
from memory.service import MemoryService
```

Update the constructor signature:

```python
def __init__(
    self,
    user_id: str = "default_user",
    session_id: str | None = None,
    memory_config: MemoryConfig | None = None,
    memory_types: list[str] | None = None,
    memory_service: MemoryServiceProtocol | None = None,
) -> None:
```

Update service creation:

```python
self.memory_service = memory_service or MemoryService(
    user_id=user_id,
    config=self.memory_config,
    memory_types=self.memory_types,
)
```

Remove direct `MemoryManager` construction from this file.

- [ ] **Step 4: Replace manager calls with service calls**

Use these replacements in `tools/builtin/memory_tool.py`:

```python
self.memory_manager.add_memory(...)       -> self.memory_service.add(...)
self.memory_manager.search_memory(...)    -> self.memory_service.search(...)
self.memory_manager.update_memory(...)    -> self.memory_service.update(...)
self.memory_manager.remove_memory(...)    -> self.memory_service.remove(...)
self.memory_manager.forget_memories(...)  -> self.memory_service.forget(...)
self.memory_manager.consolidate_working_to_episodic(...) -> self.memory_service.consolidate(...)
self.memory_manager.clear_memories(...)   -> self.memory_service.clear(...)
self.memory_manager.memory_stats(...)     -> self.memory_service.stats(...)
self.memory_manager.memory_summary(...)   -> self.memory_service.summary(...)
```

- [ ] **Step 5: Maintain backward compatibility for tests if needed**

If many tests still construct `MemoryTool(memory_manager=FakeMemoryManager())`, support a temporary deprecated parameter:

```python
memory_manager: MemoryManagerProtocol | None = None,
```

Then create the service with:

```python
if memory_service is not None:
    self.memory_service = memory_service
elif memory_manager is not None:
    self.memory_service = MemoryService(
        user_id=user_id,
        config=self.memory_config,
        memory_types=self.memory_types,
        manager=memory_manager,
    )
else:
    self.memory_service = MemoryService(
        user_id=user_id,
        config=self.memory_config,
        memory_types=self.memory_types,
    )
```

This keeps the migration small. Add a cleanup note to `docs/system-issues.md` if the deprecated parameter remains.

- [ ] **Step 6: Run MemoryTool tests**

Run:

```bash
uv run pytest tests/test_memory_tool.py -q
```

Expected: PASS.

---

### Task 4: Allow Tool Registry to Inject MemoryService

**Files:**
- Modify: `tools/agent_registry.py`
- Test: `tests/test_simple_agent_memory.py`, `tests/test_react_agent_with_tools.py`

- [ ] **Step 1: Update registry signature**

Modify `create_agent_tool_registry`:

```python
from memory.protocols import MemoryServiceProtocol
```

Add parameter:

```python
memory_service: MemoryServiceProtocol | None = None,
```

Register memory tool with:

```python
registry.register_tool(
    MemoryTool(
        user_id=memory_user_id,
        memory_types=types,
        memory_service=memory_service,
    ),
)
```

- [ ] **Step 2: Add registry test**

Add to an existing registry or memory integration test:

```python
from memory.service import MemoryService
from tests.memory_fakes import FakeMemoryManager
from tools.agent_registry import create_agent_tool_registry


def test_agent_tool_registry_accepts_memory_service() -> None:
    service = MemoryService(manager=FakeMemoryManager("registry_mem_123"))
    registry = create_agent_tool_registry(
        enable_search=False,
        enable_calculator=False,
        enable_rag=False,
        enable_memory=True,
        memory_service=service,
    )

    tool = registry.get_tool("memory")
    result = tool.execute("add", content="hello", memory_type="working")

    assert "registry" in result
```

- [ ] **Step 3: Run agent memory tests**

Run:

```bash
uv run pytest tests/test_simple_agent_memory.py tests/test_react_agent_with_tools.py tests/test_memory_tool.py -q
```

Expected: PASS.

---

### Task 5: Document the New Boundary

**Files:**
- Modify: `docs/architecture/memory.md`
- Modify: `docs/system-issues.md`

- [ ] **Step 1: Update architecture doc**

In `docs/architecture/memory.md`, replace the top diagram with:

```text
Agent Runtime / Memory Hooks
    │
    ▼
MemoryService（内部稳定 API）
    │
    ├─► MemoryTool（LLM/用户显式操作的 Tool Adapter）
    │
    ▼
MemoryManager（按 memory_types 路由 + 依赖注入）
    │
    ├─► WorkingMemory     → InMemoryStore
    ├─► EpisodicMemory    → PostgreSQL + MilvusVectorStore
    ├─► SemanticMemory    → Neo4j + MilvusVectorStore
    └─► PerceptualMemory  → 内存元数据 + 多模态 MilvusVectorStore
```

Add a short note:

```markdown
`MemoryService` 是 Agent 内部使用的记忆边界；`MemoryTool` 只是该服务的 Tool Adapter。
后续 Agent 可在 run 前调用 service 检索上下文，在 run 后调用 service 沉淀交互，不必依赖 LLM 主动调用 memory 工具。
```

- [ ] **Step 2: Update issue tracker**

In `docs/system-issues.md`:

- Mark A-02 as `mitigated` if `MemoryTool` depends on `MemoryServiceProtocol`.
- Keep A-05 open because `manager.py` is still heavy.
- Add a new architecture item:

```markdown
### A-07 · Agent Runtime Memory Hooks

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P2 |
| **问题** | `MemoryService` 已可供内部调用，但 Agent run 前检索 / run 后沉淀尚未接入。 |
| **目标方案** | 在 SimpleAgent / ReActAgent 外层加入可配置 memory hooks，而不是依赖 LLM 主动调用 MemoryTool。 |
```

- [ ] **Step 3: Run docs-sensitive tests**

Run:

```bash
uv run pytest tests/test_memory_tool.py tests/test_simple_agent_memory.py -q
```

Expected: PASS.

---

### Task 6: Full Verification

**Files:**
- No new code files.

- [ ] **Step 1: Run full test suite**

Run:

```bash
uv run pytest tests/ -q
```

Expected: `161 passed, 3 skipped` or the updated equivalent if new tests increase the pass count.

- [ ] **Step 2: Check lints for edited files**

Use Cursor lints or run:

```bash
uv run python -m compileall memory tools tests
```

Expected: no syntax errors.

- [ ] **Step 3: Inspect references**

Run:

```bash
rg "memory_manager=|MemoryManagerProtocol|MemoryServiceProtocol|MemoryService" tools tests memory docs
```

Expected:
- `MemoryService` appears in `memory/service.py`, `MemoryTool`, registry tests, and docs.
- `memory_manager=` appears only in backward-compat tests or temporary compatibility paths.
- `MemoryManagerProtocol` remains only where the service wraps the existing manager.

- [ ] **Step 4: Commit only if explicitly requested**

Do not commit automatically. If the user asks for a commit, stage only files touched by this plan and use a concise message:

```bash
git add memory/service.py memory/protocols.py tools/builtin/memory_tool.py tools/agent_registry.py tests/test_memory_service.py tests/test_memory_tool.py docs/architecture/memory.md docs/system-issues.md
git commit -m "$(cat <<'EOF'
Introduce memory service adapter boundary.

EOF
)"
```

---

## Self-Review

- **Spec coverage:** The plan covers the requested design shift from Tool-owned memory to `MemoryService + Tool Adapter`. It deliberately does not implement Agent runtime hooks; those are documented as the next architecture item.
- **Placeholder scan:** No `TBD`, `TODO`, or unspecified implementation steps remain. Every code-changing task includes concrete paths and code snippets.
- **Type consistency:** `MemoryServiceProtocol`, `MemoryManagerProtocol`, and `MemoryService` method names match across tasks. `MemoryTool` uses service verbs (`add`, `search`, `update`, `remove`, `forget`, `consolidate`, `clear`, `stats`, `summary`).

