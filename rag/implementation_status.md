# RAG 实现状态

## 已实现

- 独立 `rag/` 包：转换、分块、PostgreSQL 元数据、Milvus 向量、ingest/search/answer。
- MarkItDown + PlainText 兜底；结构感知 Markdown 分块。
- **`rag_vector_outbox`**：摄取先入队，与 memory 共用 `memory_vector_worker.py` 写 Milvus。
- `RagTool`：`ingest` / `search` / `answer` / `delete` / `reindex` / `list_documents` / `stats`。
- 查询策略：`direct`、`hyde`、`multi_query`。
- **reranker**：`NoneReranker`（默认透传）+ `LLMReranker`（LLM 打分重排序）；`RagRetriever.search(rerank=...)` 启用，候选扩大 `rerank_candidate_factor×top_k`。
- Agent 集成 + `scripts/try_rag.py`；单元测试用 Fake 后端。

## 当前妥协

- 摄取后 document 可标 indexed，chunk 在 worker 处理前 `indexed=false`。
- HyDE / multi_query 需额外 LLM 调用。
- 无 URL / 目录批量摄取。

## 待办

详见 [docs/system-issues.md](../docs/system-issues.md)（G-03 批量摄取等）。
