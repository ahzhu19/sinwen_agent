"""可选的真实 PostgreSQL + Milvus 集成测试。

运行方式：
    RUN_EPISODIC_INTEGRATION=1 uv run pytest tests/test_episodic_integration.py -v
"""

from __future__ import annotations

import os

import pytest

from memory.config import MemoryConfig
from memory.manager import MemoryManager


@pytest.mark.skipif(
    os.getenv("RUN_EPISODIC_INTEGRATION") != "1",
    reason="需要本地 PostgreSQL、Milvus 与 EMBED_API_KEY",
)
def test_episodic_memory_real_database_roundtrip() -> None:
    config = MemoryConfig.from_env()
    if not config.embed_api_key:
        pytest.skip("未配置 EMBED_API_KEY")

    manager = MemoryManager(
        config=config,
        user_id="integration_user",
        enable_working=False,
        enable_episodic=True,
        enable_semantic=False,
    )

    memory_id = manager.add_memory(
        content="集成测试：用户完成了数据库迁移",
        memory_type="episodic",
        importance=0.9,
        metadata={"session_id": "integration_session", "source": "integration_test"},
    )

    results = manager.search_memory(
        query="数据库迁移",
        memory_type="episodic",
        limit=3,
        session_id="integration_session",
    )

    assert any(record.id == memory_id for record in results)

    manager.memory_modules["episodic"].remove(memory_id)
