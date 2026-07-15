"""RAG tool for document knowledge retrieval."""

from __future__ import annotations

from typing import Any, Protocol

from rag.manager import RagManager
from rag.models import BatchIngestResult, RagAnswer, RagDocument, RagSearchResult
from tools.base import Tool

SUPPORTED_RAG_ACTIONS = (
    "ingest",
    "search",
    "answer",
    "delete",
    "reindex",
    "list_documents",
    "stats",
)


class RagManagerProtocol(Protocol):
    def ingest(
        self,
        source: str,
        source_type: str = "file",
        metadata: dict[str, Any] | None = None,
    ) -> RagDocument:
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "direct",
        rerank: str | bool | None = None,
    ) -> list[RagSearchResult]:
        ...

    def answer(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "direct",
        rerank: str | bool | None = None,
    ) -> RagAnswer:
        ...

    def list_documents(self, limit: int = 50) -> list[RagDocument]:
        ...

    def delete(self, document_id: str) -> None:
        ...

    def reindex(self, document_id: str) -> RagDocument:
        ...

    def stats(self) -> dict[str, Any]:
        ...

    def ingest_url(
        self,
        url: str,
        metadata: dict[str, Any] | None = None,
    ) -> RagDocument:
        ...

    def ingest_directory(
        self,
        path: str,
        pattern: str = "**/*.md",
        metadata: dict[str, Any] | None = None,
    ) -> BatchIngestResult:
        ...


class RagTool(Tool):
    def __init__(self, rag_manager: RagManagerProtocol | None = None) -> None:
        self.rag_manager = rag_manager or RagManager()

    @property
    def name(self) -> str:
        return "rag"

    @property
    def description(self) -> str:
        return (
            "摄取和查询外部知识文档，支持 RAG 检索与带来源回答；"
            "可删除、重建索引、列出文档与统计"
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(SUPPORTED_RAG_ACTIONS),
                    "description": "RAG 操作",
                },
                "source": {
                    "type": "string",
                    "description": "要摄取的本地文件路径",
                },
                "source_type": {
                    "type": "string",
                    "enum": ["file", "url", "directory"],
                    "description": "知识源类型：file（本地文件）、url（网页 URL）、directory（目录批量）",
                },
                "pattern": {
                    "type": "string",
                    "description": "目录摄取时的文件匹配模式（source_type=directory 时使用），默认 **/*.md",
                },
                "query": {
                    "type": "string",
                    "description": "查询问题",
                },
                "top_k": {
                    "type": "integer",
                    "description": "检索片段数量",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["direct", "hyde", "multi_query"],
                    "description": "检索策略：direct / hyde / multi_query",
                },
                "rerank": {
                    "type": "string",
                    "enum": ["none", "llm"],
                    "description": "重排序：none（默认）或 llm（LLM 打分重排序，延迟更高）",
                },
                "document_id": {
                    "type": "string",
                    "description": "文档 UUID（delete / reindex）",
                },
                "limit": {
                    "type": "integer",
                    "description": "list_documents 返回条数上限",
                },
            },
            "required": ["action"],
        }

    def run(self, **kwargs: Any) -> str:
        action = str(kwargs.pop("action", ""))
        return self.execute(action, **kwargs)

    def execute(self, action: str, **kwargs: Any) -> str:
        if action == "ingest":
            return self._ingest(**kwargs)
        if action == "search":
            return self._search(**kwargs)
        if action == "answer":
            return self._answer(**kwargs)
        if action == "delete":
            return self._delete(**kwargs)
        if action == "reindex":
            return self._reindex(**kwargs)
        if action == "list_documents":
            return self._list_documents(**kwargs)
        if action == "stats":
            return self._stats(**kwargs)
        return f"错误：不支持的 RAG 操作 '{action}'。支持的操作：{', '.join(SUPPORTED_RAG_ACTIONS)}"

    def _ingest(
        self,
        source: str = "",
        source_type: str = "file",
        pattern: str = "**/*.md",
        **metadata: Any,
    ) -> str:
        try:
            if not source.strip():
                return "❌ 摄取失败: source 不能为空"
            meta = dict(metadata)
            if source_type == "directory":
                result = self.rag_manager.ingest_directory(
                    source,
                    pattern=pattern,
                    metadata=meta,
                )
                lines = [
                    f"✅ 目录摄取完成: 成功 {result.success_count} 篇, "
                    f"失败 {result.error_count} 篇"
                ]
                if result.errors:
                    lines.append("失败文件：")
                    lines.extend(f"  - {err}" for err in result.errors)
                if result.documents:
                    lines.append("已摄取文档：")
                    for doc in result.documents:
                        lines.append(
                            f"  - [{doc.id[:8]}...] "
                            f"{doc.title or doc.source_uri}"
                        )
                return "\n".join(lines)
            if source_type == "url":
                document = self.rag_manager.ingest_url(source, metadata=meta)
                return (
                    f"✅ RAG 文档已摄取 (ID: {document.id[:8]}..., "
                    f"标题: {document.title or document.source_uri})"
                )
            document = self.rag_manager.ingest(
                source=source,
                source_type=source_type,
                metadata=meta,
            )
            return (
                f"✅ RAG 文档已摄取 (ID: {document.id[:8]}..., "
                f"标题: {document.title or document.source_uri})"
            )
        except Exception as exc:
            return f"❌ 摄取失败: {exc}"

    def _search(
        self,
        query: str = "",
        top_k: int = 5,
        strategy: str = "direct",
        rerank: str | None = None,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            if not query.strip():
                return "❌ 搜索失败: query 不能为空"
            results = self.rag_manager.search(
                query=query,
                top_k=top_k,
                strategy=strategy,
                rerank=rerank,
            )
            if not results:
                return f"未找到与「{query}」相关的知识片段"
            lines = [f"找到 {len(results)} 个相关片段（策略: {strategy}）："]
            for index, result in enumerate(results, start=1):
                heading = " / ".join(result.chunk.heading_path) or "(无标题)"
                lines.append(
                    f"{index}. [{result.score:.3f}] "
                    f"{result.document.title or result.document.source_uri} - {heading}\n"
                    f"   {result.chunk.content[:200]}"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ 搜索失败: {exc}"

    def _answer(
        self,
        query: str = "",
        top_k: int = 5,
        strategy: str = "direct",
        rerank: str | None = None,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        try:
            if not query.strip():
                return "❌ 回答失败: query 不能为空"
            answer = self.rag_manager.answer(
                query=query,
                top_k=top_k,
                strategy=strategy,
                rerank=rerank,
            )
            lines = [answer.answer]
            if answer.sources:
                lines.append("\n来源：")
                for index, result in enumerate(answer.sources, start=1):
                    heading = " / ".join(result.chunk.heading_path) or "(无标题)"
                    lines.append(
                        f"[Source {index}] "
                        f"{result.document.title or result.document.source_uri} - {heading}"
                    )
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ 回答失败: {exc}"

    def _delete(self, document_id: str = "", **kwargs: Any) -> str:
        _ = kwargs
        try:
            if not document_id.strip():
                return "❌ 删除失败: document_id 不能为空"
            self.rag_manager.delete(document_id.strip())
            return f"✅ 已删除文档 {document_id[:8]}..."
        except Exception as exc:
            return f"❌ 删除失败: {exc}"

    def _reindex(self, document_id: str = "", **kwargs: Any) -> str:
        _ = kwargs
        try:
            if not document_id.strip():
                return "❌ 重建索引失败: document_id 不能为空"
            document = self.rag_manager.reindex(document_id.strip())
            return (
                f"✅ 已重建索引 (ID: {document.id[:8]}..., "
                f"状态: {document.status})"
            )
        except Exception as exc:
            return f"❌ 重建索引失败: {exc}"

    def _list_documents(self, limit: int = 50, **kwargs: Any) -> str:
        _ = kwargs
        try:
            documents = self.rag_manager.list_documents(limit=limit)
            if not documents:
                return "知识库中暂无文档"
            lines = [f"共 {len(documents)} 篇文档："]
            for index, document in enumerate(documents, start=1):
                title = document.title or document.source_uri
                lines.append(
                    f"{index}. [{document.status}] {title} "
                    f"(id={document.id[:8]}...)"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"❌ 列表失败: {exc}"

    def _stats(self, **kwargs: Any) -> str:
        _ = kwargs
        try:
            stats = self.rag_manager.stats()
            return (
                f"RAG 统计: 文档 {stats['document_count']} 篇, "
                f"chunk {stats['chunk_count']} 个 "
                f"(已索引 {stats['indexed_chunk_count']}), "
                f"Milvus 集合 {stats['collection']}"
            )
        except Exception as exc:
            return f"❌ 统计失败: {exc}"
