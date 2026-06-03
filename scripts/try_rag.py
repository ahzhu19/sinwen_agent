"""RAG 知识库真机试用脚本。

在项目根目录运行：

    uv run python scripts/try_rag.py ingest --source docs/README.md
    uv run python scripts/try_rag.py search --query "项目结构"
    uv run python scripts/try_rag.py answer --query "RAG 如何工作？"
    uv run python scripts/try_rag.py list
    uv run python scripts/try_rag.py stats
    uv run python scripts/try_rag.py --agent --task "列出知识库里的文档"

需要 `.env` 中配置 DATABASE_URL、MILVUS_*、EMBED_* 以及 LLM 相关变量。
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM
from rag.manager import RagManager
from tools.agent_registry import create_agent_tool_registry
from tools.builtin.rag_tool import RagTool


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
    from rag.config import RagConfig

    config = RagConfig.from_env()
    postgres_url = urlparse(config.database_url or "")
    check_tcp_service(
        "PostgreSQL",
        postgres_url.hostname or "localhost",
        postgres_url.port or 5432,
    )
    milvus_host = (config.milvus_uri or "").replace("http://", "").split(":")[0] or "localhost"
    check_tcp_service("Milvus", milvus_host, 19530)


def run_ingest(source: str) -> None:
    manager = RagManager()
    document = manager.ingest(source=source)
    print(
        f"✅ 已摄取: id={document.id}\n"
        f"   标题: {document.title or document.source_uri}\n"
        f"   状态: {document.status}"
    )


def run_search(query: str, top_k: int, strategy: str) -> None:
    tool = RagTool(RagManager())
    print(tool.execute("search", query=query, top_k=top_k, strategy=strategy))


def run_answer(query: str, top_k: int, strategy: str) -> None:
    tool = RagTool(RagManager())
    print(tool.execute("answer", query=query, top_k=top_k, strategy=strategy))


def run_list(limit: int) -> None:
    tool = RagTool(RagManager())
    print(tool.execute("list_documents", limit=limit))


def run_stats() -> None:
    tool = RagTool(RagManager())
    print(tool.execute("stats"))


def run_delete(document_id: str) -> None:
    tool = RagTool(RagManager())
    print(tool.execute("delete", document_id=document_id))


def run_reindex(document_id: str) -> None:
    tool = RagTool(RagManager())
    print(tool.execute("reindex", document_id=document_id))


def run_agent(task: str) -> None:
    llm = BaseLLM()
    registry = create_agent_tool_registry(
        enable_search=False,
        enable_calculator=False,
        enable_rag=True,
    )
    agent = SimpleAgent(
        name="RAG 助手",
        llm=llm,
        tool_registry=registry,
        enable_tool_calling=True,
        max_tool_iterations=5,
    )
    print(f"已注册工具: {agent.list_tools()}\n")
    print(f"用户问题: {task}\n")
    print(agent.run(task))


def main() -> None:
    parser = argparse.ArgumentParser(description="试用 RAG 知识库")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="摄取本地文件")
    ingest_parser.add_argument("--source", required=True, help="文件路径")

    search_parser = subparsers.add_parser("search", help="向量检索")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument(
        "--strategy",
        default="direct",
        choices=["direct", "hyde", "multi_query"],
    )

    answer_parser = subparsers.add_parser("answer", help="检索并生成回答")
    answer_parser.add_argument("--query", required=True)
    answer_parser.add_argument("--top-k", type=int, default=5)
    answer_parser.add_argument(
        "--strategy",
        default="direct",
        choices=["direct", "hyde", "multi_query"],
    )

    list_parser = subparsers.add_parser("list", help="列出已摄取文档")
    list_parser.add_argument("--limit", type=int, default=20)

    subparsers.add_parser("stats", help="知识库统计")
    delete_parser = subparsers.add_parser("delete", help="删除文档")
    delete_parser.add_argument("--document-id", required=True)
    reindex_parser = subparsers.add_parser("reindex", help="重建向量索引")
    reindex_parser.add_argument("--document-id", required=True)

    parser.add_argument(
        "--agent",
        action="store_true",
        help="通过 SimpleAgent + rag 工具对话",
    )
    parser.add_argument(
        "--task",
        default="请列出知识库中的文档，并简要说明你能用 rag 工具做什么",
        help="SimpleAgent 模式下的用户问题",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="跳过 PostgreSQL / Milvus 连通性检查",
    )

    args = parser.parse_args()

    try:
        if not args.skip_check:
            check_services()

        if args.agent:
            run_agent(args.task)
            return

        if args.command == "ingest":
            run_ingest(args.source)
        elif args.command == "search":
            run_search(args.query, args.top_k, args.strategy)
        elif args.command == "answer":
            run_answer(args.query, args.top_k, args.strategy)
        elif args.command == "list":
            run_list(args.limit)
        elif args.command == "stats":
            run_stats()
        elif args.command == "delete":
            run_delete(args.document_id)
        elif args.command == "reindex":
            run_reindex(args.document_id)
        else:
            parser.print_help()
            raise SystemExit(1)
    except Exception as exc:
        print(f"\n❌ 试用失败：{exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
