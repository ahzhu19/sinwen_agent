"""Milvus 向量 outbox Worker：处理 Postgres memory_vector_outbox 与 Neo4j 语义 outbox。

用法（项目根目录）：

    uv run python scripts/memory_vector_worker.py
    uv run python scripts/memory_vector_worker.py --loop --interval 10
    uv run python scripts/memory_vector_worker.py --batch-size 50 --once
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.config import MemoryConfig
from memory.semantic_outbox_processor import SemanticOutboxProcessor
from memory.storage.milvus_store import create_vector_store
from memory.storage.neo4j_store import create_semantic_store
from memory.storage.postgres_outbox_store import create_postgres_outbox_store
from memory.vector_outbox_processor import VectorOutboxProcessor


def run_once(*, batch_size: int) -> dict[str, tuple[int, int]]:
    config = MemoryConfig.from_env()
    results: dict[str, tuple[int, int]] = {}

    if config.database_url:
        outbox = create_postgres_outbox_store(config)
        processor = VectorOutboxProcessor(config, outbox)
        pending_before = outbox.pending_count()
        for kind in ("episodic", "perceptual"):
            results[kind] = processor.process_batch(batch_size=batch_size, memory_kind=kind)
        pending_after = outbox.pending_count()
        print(f"postgres outbox pending: {pending_before} -> {pending_after}")
        for kind, (ok_count, fail_count) in results.items():
            print(f"  {kind}: 成功 {ok_count}，失败 {fail_count}")
    else:
        print("未配置 DATABASE_URL，跳过 Postgres outbox")

    if config.neo4j_password:
        semantic_store = create_semantic_store(config)
        semantic_vectors = create_vector_store(
            config,
            collection_name=config.milvus_semantic_collection,
        )
        semantic_processor = SemanticOutboxProcessor(
            config,
            semantic_store,
            vector_store=semantic_vectors,
        )
        pending_semantic = semantic_store.pending_outbox_count()
        results["semantic"] = semantic_processor.process_batch(batch_size=batch_size)
        pending_semantic_after = semantic_store.pending_outbox_count()
        ok_count, fail_count = results["semantic"]
        print(
            f"neo4j semantic outbox pending: {pending_semantic} -> {pending_semantic_after} "
            f"(成功 {ok_count}，失败 {fail_count})"
        )
    else:
        print("未配置 NEO4J_PASSWORD，跳过 Neo4j 语义 outbox")

    return results


def main() -> None:
    load_dotenv(_ROOT / ".env")
    parser = argparse.ArgumentParser(description="处理 memory_vector_outbox 并写入 Milvus")
    parser.add_argument("--once", action="store_true", help="只处理一批后退出")
    parser.add_argument("--loop", action="store_true", help="循环处理")
    parser.add_argument("--interval", type=float, default=10.0, help="循环间隔秒数")
    parser.add_argument("--batch-size", type=int, default=None, help="每批 claim 条数")
    args = parser.parse_args()

    config = MemoryConfig.from_env()
    batch_size = args.batch_size or config.vector_outbox_worker_batch_size

    try:
        if args.loop:
            while True:
                run_once(batch_size=batch_size)
                time.sleep(max(1.0, args.interval))
        else:
            run_once(batch_size=batch_size)
    except KeyboardInterrupt:
        print("\n已停止 worker")
    except Exception as exc:
        print(f"\n❌ worker 失败: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
