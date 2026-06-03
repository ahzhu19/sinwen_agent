# RAG 实现状态

## 已实现

- 独立 `rag/` 包：`config`、`models`、`converter`、`chunker`、`storage`、`vector_store`、`ingestion`、`retriever`、`generator`、`manager`、`query_strategy`。
- MarkItDown 文档转换（`MarkItDownConverter`）与纯文本/Markdown 兜底（`PlainTextConverter`）。
- 结构感知 Markdown 分块：标题路径、token 预算、overlap。
- PostgreSQL 元数据：`rag_documents`、`rag_chunks`、`rag_ingestion_runs`。
- Milvus 向量索引：`hello_agents_rag_chunks`。
- `RagTool`：`ingest` / `search` / `answer` / `delete` / `reindex` / `list_documents` / `stats`。
- 查询策略：`direct`、`hyde`、`multi_query`（`search` / `answer` 的 `strategy` 参数）。
- Agent 集成：`create_agent_tool_registry`、`SimpleAgent.with_agent_tools()`。
- 真机脚本：`scripts/try_rag.py`。
- 单元测试使用 Fake 后端，不依赖 Docker。

## 当前妥协

- 无 reranker；多策略检索按 chunk_id 取最高分合并。
- `RagManager` 默认实例化会连接真实 PostgreSQL/Milvus/Embedding，测试需注入 Fake。
- 摄取失败时 Milvus 与 Postgres 非严格事务（与 memory 模块相同）。
- HyDE / multi_query 依赖额外 LLM 调用，真机需配置 LLM API。

## 后续建议

- 接入 CrossEncoder 或 LLM rerank。
- 为摄取双写增加 outbox 或补偿任务。
- 支持 URL / 目录批量摄取。
