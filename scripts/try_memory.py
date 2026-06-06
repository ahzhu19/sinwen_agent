"""真实 Docker 记忆系统试用脚本。

在项目根目录运行：

    uv run python scripts/try_memory.py

默认验证 working / episodic / semantic 三类记忆。Perceptual（experimental）需加 ``--with-perceptual``。
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.config import MemoryConfig
from memory.manager import MemoryManager
from memory.modules import MemoryRecord


def check_tcp_service(name: str, host: str, port: int, timeout: float = 2.0) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print(f"✅ {name} 可连接：{host}:{port}")
    except OSError as exc:
        raise RuntimeError(
            f"{name} 无法连接：{host}:{port}。请先运行 `docker compose up -d`。"
        ) from exc


def print_records(title: str, records: Iterable[MemoryRecord]) -> None:
    print(f"\n{title}")
    records = list(records)
    if not records:
        print("  未找到结果")
        return
    for index, record in enumerate(records, start=1):
        print(f"  {index}. [{record.memory_type}:{record.id[:8]}] {record.content}")
        interesting = {
            key: value
            for key, value in record.metadata.items()
            if key in {"session_id", "concepts", "modality", "raw_data", "timestamp"}
        }
        if interesting:
            print(f"     metadata={interesting}")


def add_sample_memories(
    manager: MemoryManager,
    session_id: str,
    *,
    include_perceptual: bool,
) -> dict[str, str]:
    ids: dict[str, str] = {}

    ids["working"] = manager.add_memory(
        content="当前任务：验证本地 Docker 记忆系统是否能正常写入和检索",
        memory_type="working",
        importance=0.6,
        metadata={"session_id": session_id, "source": "try_memory"},
    )

    ids["episodic"] = manager.add_memory(
        content="用户今天完成了 PostgreSQL、Neo4j、Milvus 的本地记忆基础设施搭建",
        memory_type="episodic",
        importance=0.8,
        metadata={"session_id": session_id, "source": "try_memory"},
    )

    ids["semantic"] = manager.add_memory(
        content="语义记忆使用 Neo4j 保存概念关系，并使用 Milvus 做向量检索",
        memory_type="semantic",
        importance=0.9,
        metadata={
            "session_id": session_id,
            "source": "try_memory",
            "concepts": ["语义记忆", "Neo4j", "Milvus", "向量检索"],
        },
    )

    if include_perceptual:
        ids["perceptual"] = manager.add_memory(
            content="用户上传了一张记忆系统架构图",
            memory_type="perceptual",
            importance=0.7,
            metadata={
                "session_id": session_id,
                "source": "try_memory",
                "modality": "image",
                "raw_data": "/tmp/memory-architecture.png",
                "caption": "PostgreSQL Neo4j Milvus 记忆系统架构图",
            },
        )

    return ids


def run_demo(*, cleanup: bool, with_perceptual: bool) -> None:
    load_dotenv(_ROOT / ".env")
    config = MemoryConfig.from_env()

    print("🔎 检查本地 Docker 服务连通性")
    postgres_url = urlparse(config.database_url or "")
    check_tcp_service(
        "PostgreSQL",
        postgres_url.hostname or "localhost",
        postgres_url.port or 5432,
    )
    check_tcp_service("Neo4j Bolt", "localhost", 7687)
    check_tcp_service("Milvus", "localhost", 19530)

    print("\n🧠 初始化真实 MemoryManager")
    manager = MemoryManager(
        config=config,
        user_id="try_memory_user",
        enable_working=True,
        enable_episodic=True,
        enable_semantic=True,
        enable_perceptual=with_perceptual,
    )

    session_id = "try_memory_session"
    type_label = "四类" if with_perceptual else "三类"
    print(f"\n✍️ 写入{type_label}记忆")
    memory_ids = add_sample_memories(
        manager,
        session_id,
        include_perceptual=with_perceptual,
    )
    for memory_type, memory_id in memory_ids.items():
        print(f"  ✅ {memory_type}: {memory_id}")

    print_records(
        "🔍 WorkingMemory 检索：Docker 记忆系统",
        manager.search_memory(
            query="Docker 记忆系统",
            memory_type="working",
            limit=3,
            session_id=session_id,
        ),
    )
    print_records(
        "🔍 EpisodicMemory 检索：本地基础设施",
        manager.search_memory(
            query="本地 PostgreSQL Neo4j Milvus 基础设施",
            memory_type="episodic",
            limit=3,
            session_id=session_id,
        ),
    )
    print_records(
        "🔍 SemanticMemory 检索：语义记忆向量检索",
        manager.search_memory(
            query="语义记忆如何结合 Neo4j 和 Milvus",
            memory_type="semantic",
            limit=3,
            session_id=session_id,
        ),
    )
    if with_perceptual:
        print_records(
            "🔍 PerceptualMemory 检索：架构图",
            manager.search_memory(
                query="记忆系统架构图",
                memory_type="perceptual",
                limit=3,
                session_id=session_id,
            ),
        )

    episodic_module = manager.memory_modules["episodic"]
    print_records(
        "🕒 EpisodicMemory 时间线",
        episodic_module.list_timeline(session_id=session_id, limit=5),
    )

    if cleanup:
        print("\n🧹 清理本次写入的数据")
        for memory_type, memory_id in memory_ids.items():
            manager.memory_modules[memory_type].remove(memory_id)
            print(f"  ✅ 已删除 {memory_type}: {memory_id[:8]}")
    else:
        print("\nℹ️ 本次数据保留在真实后端中。若想试完即删，可加参数：--cleanup")
        if with_perceptual:
            print(
                "   注意：PerceptualMemory 元数据为进程内存 store，"
                "仅在本次脚本运行中可检索。"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="试用真实 Docker 记忆系统")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="脚本结束前删除本次写入的记忆",
    )
    parser.add_argument(
        "--with-perceptual",
        action="store_true",
        help="同时试用 PerceptualMemory（experimental）",
    )
    args = parser.parse_args()

    try:
        run_demo(cleanup=args.cleanup, with_perceptual=args.with_perceptual)
    except Exception as exc:
        print(f"\n❌ 试用失败：{exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
