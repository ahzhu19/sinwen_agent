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
from memory.outbox_status import collect_outbox_status, format_outbox_status
from memory.storage.neo4j_store import create_semantic_store
from memory.storage.postgres_outbox_store import create_postgres_outbox_store


def main() -> None:
    load_dotenv(_ROOT / ".env")
    config = MemoryConfig.from_env()

    pg_outbox = create_postgres_outbox_store(config) if config.database_url else None
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


if __name__ == "__main__":
    main()
