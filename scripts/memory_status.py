"""查看记忆系统 outbox 积压状态。

用法（项目根目录）：

    uv run python scripts/memory_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.config import MemoryConfig
from memory.outbox_maintenance import run_memory_outbox_maintenance
from memory.outbox_status import collect_outbox_status, format_outbox_status
from memory.storage.neo4j_store import create_semantic_store
from memory.storage.postgres_outbox_store import create_postgres_outbox_store
from memory.storage.postgres_store import create_episodic_store
from rag.config import RagConfig
from rag.outbox_store import create_rag_outbox_store


def main() -> None:
    load_dotenv(_ROOT / ".env")
    config = MemoryConfig.from_env()
    rag_config = RagConfig.from_env()

    pg_outbox = create_postgres_outbox_store(config) if config.database_url else None
    episodic_store = create_episodic_store(config) if config.database_url else None
    semantic_store = (
        create_semantic_store(config)
        if config.neo4j_password
        else None
    )

    status = collect_outbox_status(
        pg_outbox=pg_outbox,
        semantic_store=semantic_store,
    )
    print(format_outbox_status(status))

    if rag_config.enable_rag_vector_outbox and rag_config.database_url:
        rag_outbox = create_rag_outbox_store(rag_config)
        rag_counts = rag_outbox.status_counts()
        print(
            "RAG outbox: "
            f"pending={rag_counts.get('pending', 0)} "
            f"processing={rag_counts.get('processing', 0)} "
            f"dead={rag_counts.get('dead', 0)}"
        )

    maintenance = run_memory_outbox_maintenance(
        config,
        pg_outbox=pg_outbox,
        episodic_store=episodic_store,
        semantic_store=semantic_store,
        reclaim_stale=False,
        replay_dead=False,
        reconcile_semantic=False,
    )
    if maintenance.get("episodic_unindexed", 0):
        print(f"Episodic 未标记 vector_indexed_at: {maintenance['episodic_unindexed']}")


if __name__ == "__main__":
    main()
