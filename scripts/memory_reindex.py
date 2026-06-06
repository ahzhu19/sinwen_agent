"""Embedding 模型迁移：为 stale 记忆重新入队 outbox。

用法（项目根目录）：

    uv run python scripts/memory_reindex.py --dry-run
    uv run python scripts/memory_reindex.py --batch-size 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.config import MemoryConfig
from memory.embedding import create_embedding_provider
from memory.storage.neo4j_store import create_semantic_store
from memory.storage.postgres_store import create_episodic_store


def reindex_episodic(
    config: MemoryConfig,
    *,
    batch_size: int,
    dry_run: bool,
) -> int:
    if not config.database_url:
        print("未配置 DATABASE_URL，跳过 episodic reindex")
        return 0

    store = create_episodic_store(config)
    embeddings = create_embedding_provider(config)
    collection_name = config.episodic_milvus_collection()
    target_model = config.embed_model_name

    if not hasattr(store, "list_stale_embeddings"):
        print("episodic store 不支持 list_stale_embeddings，跳过")
        return 0

    stale = store.list_stale_embeddings(embedding_model=target_model, limit=batch_size)
    if dry_run:
        print(f"[dry-run] episodic stale: {len(stale)} 条 (model != {target_model})")
        return len(stale)

    reindexed = 0
    for event in stale:
        vector = embeddings.embed(event.content)
        store.update_with_vector_outbox(
            memory_id=event.id,
            user_id=event.user_id,
            content=event.content,
            importance=event.importance,
            metadata=dict(event.metadata),
            vector=vector,
            collection_name=collection_name,
            max_attempts=config.vector_outbox_max_attempts,
            embedding_model=target_model,
            session_id=event.session_id,
        )
        reindexed += 1
    print(f"episodic reindex 入队: {reindexed} 条 -> {collection_name}")
    return reindexed


def reindex_semantic(
    config: MemoryConfig,
    *,
    batch_size: int,
    dry_run: bool,
) -> int:
    if not config.neo4j_password:
        print("未配置 NEO4J_PASSWORD，跳过 semantic reindex")
        return 0

    store = create_semantic_store(config)
    if not hasattr(store, "ensure_pending_outbox_events"):
        print("semantic store 不支持 ensure_pending_outbox_events，跳过")
        return 0

    if dry_run:
        pending = store.pending_outbox_count() if hasattr(store, "pending_outbox_count") else 0
        print(f"[dry-run] semantic pending outbox: {pending}")
        return pending

    created = store.ensure_pending_outbox_events(
        batch_size=batch_size,
        max_attempts=config.vector_outbox_max_attempts,
        collection_name=config.semantic_milvus_collection(),
    )
    print(f"semantic 对账补 outbox: {created} 条")
    return created


def main() -> None:
    load_dotenv(_ROOT / ".env")
    parser = argparse.ArgumentParser(description="为 stale embedding 记忆重新入队 outbox")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--episodic-only", action="store_true")
    parser.add_argument("--semantic-only", action="store_true")
    args = parser.parse_args()

    config = MemoryConfig.from_env()
    print(f"目标 embedding: {config.embed_model_name} dim={config.milvus_vector_size}")
    print(f"episodic collection: {config.episodic_milvus_collection()}")
    print(f"semantic collection: {config.semantic_milvus_collection()}")

    if not args.semantic_only:
        reindex_episodic(config, batch_size=args.batch_size, dry_run=args.dry_run)
    if not args.episodic_only:
        reindex_semantic(config, batch_size=args.batch_size, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
