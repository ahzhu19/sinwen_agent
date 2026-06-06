"""Milvus 向量 outbox Worker：处理 Postgres memory_vector_outbox、Neo4j 语义 outbox 与 RAG outbox。

用法（项目根目录）：

    uv run python scripts/memory_vector_worker.py
    uv run python scripts/memory_vector_worker.py --loop --interval 10
    uv run python scripts/memory_vector_worker.py --batch-size 50 --once
    uv run python scripts/memory_vector_worker.py --replay-dead --once
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
from memory.outbox_maintenance import run_memory_outbox_maintenance
from memory.semantic_outbox_processor import SemanticOutboxProcessor
from memory.storage.milvus_store import create_vector_store
from memory.storage.neo4j_store import create_semantic_store
from memory.storage.postgres_outbox_store import create_postgres_outbox_store
from memory.storage.postgres_store import create_episodic_store
from memory.vector_outbox_processor import VectorOutboxProcessor
from rag.config import RagConfig
from rag.outbox_processor import RagVectorOutboxProcessor
from rag.outbox_store import create_rag_outbox_store
from rag.storage import create_rag_store
from rag.vector_store import MilvusRagVectorStore


def run_once(
    *,
    batch_size: int,
    replay_dead: bool = False,
    reconcile_semantic: bool = True,
) -> dict[str, tuple[int, int] | int]:
    config = MemoryConfig.from_env()
    rag_config = RagConfig.from_env()
    results: dict[str, tuple[int, int] | int] = {}

    pg_outbox = None
    episodic_store = None
    semantic_store = None

    if config.database_url:
        pg_outbox = create_postgres_outbox_store(config)
        episodic_store = create_episodic_store(config)

    if config.neo4j_password:
        semantic_store = create_semantic_store(config)

    maintenance = run_memory_outbox_maintenance(
        config,
        pg_outbox=pg_outbox,
        episodic_store=episodic_store,
        semantic_store=semantic_store,
        reclaim_stale=True,
        replay_dead=replay_dead,
        reconcile_semantic=reconcile_semantic,
    )
    if maintenance:
        print(f"outbox maintenance: {maintenance}")

    if config.database_url and pg_outbox is not None and episodic_store is not None:
        processor = VectorOutboxProcessor(
            config,
            pg_outbox,
            episodic_store=episodic_store,
        )
        pending_before = pg_outbox.pending_count()
        batch_results: dict[str, tuple[int, int]] = {}
        for kind in ("episodic", "perceptual"):
            batch_results[kind] = processor.process_batch(
                batch_size=batch_size,
                memory_kind=kind,
                reclaim_stale=False,
            )
        pending_after = pg_outbox.pending_count()
        print(f"postgres outbox pending: {pending_before} -> {pending_after}")
        for kind, (ok_count, fail_count) in batch_results.items():
            print(f"  {kind}: 成功 {ok_count}，失败 {fail_count}")
        results.update(batch_results)
    else:
        print("未配置 DATABASE_URL，跳过 Postgres outbox")

    if semantic_store is not None:
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
        results["semantic"] = semantic_processor.process_batch(
            batch_size=batch_size,
            reclaim_stale=False,
        )
        pending_semantic_after = semantic_store.pending_outbox_count()
        ok_count, fail_count = results["semantic"]
        print(
            f"neo4j semantic outbox pending: {pending_semantic} -> {pending_semantic_after} "
            f"(成功 {ok_count}，失败 {fail_count})"
        )
    else:
        print("未配置 NEO4J_PASSWORD，跳过 Neo4j 语义 outbox")

    if rag_config.enable_rag_vector_outbox and rag_config.database_url:
        rag_outbox = create_rag_outbox_store(rag_config)
        rag_store = create_rag_store(rag_config.database_url)
        rag_vectors = MilvusRagVectorStore(
            uri=rag_config.milvus_uri,
            collection_name=rag_config.collection_name,
            metric_type=rag_config.metric_type,
            timeout=rag_config.timeout,
        )
        rag_processor = RagVectorOutboxProcessor(
            rag_config,
            rag_outbox,
            vector_store=rag_vectors,
            rag_store=rag_store,
        )
        pending_rag = rag_outbox.pending_count()
        results["rag"] = rag_processor.process_batch(batch_size=batch_size, reclaim_stale=False)
        pending_rag_after = rag_outbox.pending_count()
        ok_count, fail_count = results["rag"]
        print(
            f"rag outbox pending: {pending_rag} -> {pending_rag_after} "
            f"(成功 {ok_count}，失败 {fail_count})"
        )

    return results


def main() -> None:
    load_dotenv(_ROOT / ".env")
    parser = argparse.ArgumentParser(description="处理 vector outbox 并写入 Milvus")
    parser.add_argument("--once", action="store_true", help="只处理一批后退出")
    parser.add_argument("--loop", action="store_true", help="循环处理")
    parser.add_argument("--interval", type=float, default=10.0, help="循环间隔秒数")
    parser.add_argument("--batch-size", type=int, default=None, help="每批 claim 条数")
    parser.add_argument("--replay-dead", action="store_true", help="重放 dead 条目后再处理")
    parser.add_argument(
        "--no-reconcile-semantic",
        action="store_true",
        help="跳过语义记忆对账补 outbox",
    )
    args = parser.parse_args()

    config = MemoryConfig.from_env()
    batch_size = args.batch_size or config.vector_outbox_worker_batch_size

    try:
        if args.loop:
            while True:
                run_once(
                    batch_size=batch_size,
                    replay_dead=args.replay_dead,
                    reconcile_semantic=not args.no_reconcile_semantic,
                )
                time.sleep(max(1.0, args.interval))
        else:
            run_once(
                batch_size=batch_size,
                replay_dead=args.replay_dead,
                reconcile_semantic=not args.no_reconcile_semantic,
            )
    except KeyboardInterrupt:
        print("\n已停止 worker")
    except Exception as exc:
        print(f"\n❌ worker 失败: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
