"""ContextBuilder 真机试用脚本。

在项目根目录运行：

    # 完整演示（记忆 + RAG + 对话历史 → 六分区上下文）
    uv run python scripts/try_context.py

    # 跳过 Docker 连通性检查（服务已确认可用时）
    uv run python scripts/try_context.py --skip-check

    # 试完删除本次写入的记忆
    uv run python scripts/try_context.py --cleanup

依赖：
    - Docker：`docker compose up -d`（PostgreSQL / Neo4j / Milvus / memory-worker）
    - `.env` 中配置 DATABASE_URL、NEO4J_*、MILVUS_*、EMBED_* 等
"""

from __future__ import annotations

import argparse
import socket
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from context import ContextBuilder, ContextConfig
from core.message import Message
from memory.config import MemoryConfig
from memory.manager import MemoryManager
from rag.manager import RagManager
from tools.builtin.memory_tool import MemoryTool
from tools.builtin.rag_tool import RagTool

USER_ID = "try_context_user"
SESSION_ID = "try_context_session"
DEFAULT_RAG_SOURCE = _ROOT / "docs" / "architecture" / "memory.md"


def check_tcp_service(name: str, host: str, port: int, timeout: float = 2.0) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print(f"✅ {name} 可连接：{host}:{port}")
    except OSError as exc:
        raise RuntimeError(
            f"{name} 无法连接：{host}:{port}。请先运行 `docker compose up -d`。"
        ) from exc


def check_services() -> None:
    load_dotenv(_ROOT / ".env")
    config = MemoryConfig.from_env()
    postgres_url = urlparse(config.database_url or "")
    check_tcp_service(
        "PostgreSQL",
        postgres_url.hostname or "localhost",
        postgres_url.port or 55432,
    )
    check_tcp_service("Neo4j Bolt", "localhost", 7687)
    check_tcp_service("Milvus", "localhost", 19530)


def _build_memory_tool() -> MemoryTool:
    config = MemoryConfig.from_env()
    manager = MemoryManager(
        config=config,
        user_id=USER_ID,
        enable_working=True,
        enable_episodic=True,
        enable_semantic=True,
    )
    return MemoryTool(
        user_id=USER_ID,
        session_id=SESSION_ID,
        memory_manager=manager,
        memory_types=["working", "episodic", "semantic"],
    )


def _ensure_rag_document(rag_manager: RagManager, source: Path) -> None:
    if not source.exists():
        print(f"⚠️ RAG 源文件不存在，跳过摄取：{source}")
        return

    documents = rag_manager.list_documents(limit=50)
    source_uri = str(source.resolve())
    if any(doc.source_uri == source_uri for doc in documents):
        print(f"ℹ️ RAG 文档已存在，跳过摄取：{source.name}")
        return

    print(f"📥 摄取 RAG 文档：{source}")
    document = rag_manager.ingest(source=str(source), source_type="file")
    print(f"   ✅ id={document.id[:8]}... 状态={document.status}")


def _seed_memories(memory_tool: MemoryTool) -> list[tuple[str, str]]:
    """写入示例记忆，返回 (memory_id, memory_type) 列表供 cleanup 使用。"""
    samples = [
        {
            "content": "用户正在开发数据分析工具,使用Python和Pandas",
            "memory_type": "semantic",
            "importance": 0.8,
        },
        {
            "content": "已完成CSV读取模块的开发",
            "memory_type": "episodic",
            "importance": 0.7,
        },
        {
            "content": "当前会话关注 Pandas 内存优化与大数据处理",
            "memory_type": "working",
            "importance": 0.6,
        },
    ]
    written: list[tuple[str, str]] = []
    print("\n✍️ 写入示例记忆")
    for sample in samples:
        memory_id = memory_tool.memory_service.add(
            content=sample["content"],
            memory_type=sample["memory_type"],
            importance=sample["importance"],
            metadata={"session_id": SESSION_ID, "source": "try_context"},
        )
        written.append((memory_id, sample["memory_type"]))
        print(f"   ✅ {sample['memory_type']}: {memory_id[:8]}...")
    return written


def _conversation_history() -> list[Message]:
    base = datetime.now()
    return [
        Message(
            content="我正在开发一个数据分析工具",
            role="user",
            timestamp=base,
        ),
        Message(
            content="很好!数据分析工具通常需要处理大量数据。您计划使用什么技术栈?",
            role="assistant",
            timestamp=base,
        ),
        Message(
            content="我打算使用Python和Pandas,已经完成了CSV读取模块",
            role="user",
            timestamp=base,
        ),
        Message(
            content="不错的选择!Pandas在数据处理方面非常强大。接下来您可能需要考虑数据清洗和转换。",
            role="assistant",
            timestamp=base,
        ),
    ]


def run_demo(*, cleanup: bool, rag_source: Path) -> None:
    load_dotenv(_ROOT / ".env")

    print("🧠 初始化 MemoryTool + RagTool")
    memory_tool = _build_memory_tool()
    rag_manager = RagManager()
    rag_tool = RagTool(rag_manager=rag_manager)

    _ensure_rag_document(rag_manager, rag_source)
    memory_ids = _seed_memories(memory_tool)

    config = ContextConfig(
        max_tokens=3000,
        reserve_ratio=0.2,
        min_relevance=0.1,
        enable_compression=True,
    )
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=config,
    )

    print("\n🔧 构建上下文")
    result = builder.build(
        user_query="如何优化Pandas的内存占用?",
        conversation_history=_conversation_history(),
        system_instructions=(
            "你是一位资深的Python数据工程顾问。你的回答需要:"
            "1) 提供具体可行的建议 2) 解释技术原理 3) 给出代码示例"
        ),
        session_id=SESSION_ID,
    )

    print("\n" + "=" * 80)
    print("构建的上下文 (result.text)")
    print("=" * 80)
    print(result.text)
    print("=" * 80)

    print("\n📊 统计 (result.stats)")
    for key, value in result.stats.items():
        print(f"   {key}: {value}")

    print(f"\n💬 LLM 消息数: {len(result.messages)}")
    print(f"   首条 role: {result.messages[0]['role']}")

    if cleanup and memory_ids:
        print("\n🧹 清理本次写入的记忆")
        for memory_id, memory_type in memory_ids:
            memory_tool.run(
                action="remove",
                memory_id=memory_id,
                memory_type=memory_type,
            )
            print(f"   ✅ 已删除 {memory_type}: {memory_id[:8]}...")
        print("   RAG 文档保留在知识库中")
    elif memory_ids:
        print("\nℹ️ 记忆数据已保留。试完即删请加：--cleanup")


def main() -> None:
    parser = argparse.ArgumentParser(description="试用 ContextBuilder（真实 Docker 环境）")
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="跳过 PostgreSQL / Neo4j / Milvus 连通性检查",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="脚本结束前删除本次写入的记忆",
    )
    parser.add_argument(
        "--rag-source",
        type=Path,
        default=DEFAULT_RAG_SOURCE,
        help=f"RAG 摄取源文件，默认 {DEFAULT_RAG_SOURCE.relative_to(_ROOT)}",
    )
    args = parser.parse_args()

    try:
        if not args.skip_check:
            print("🔎 检查本地 Docker 服务连通性")
            check_services()
        run_demo(cleanup=args.cleanup, rag_source=args.rag_source)
    except Exception as exc:
        print(f"\n❌ 试用失败：{exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
