# 记忆系统实现状态

## 已实现

- `MemoryTool`：9 个 action（`add` / `search` / `summary` / `stats` / `update` / `remove` / `forget` / `consolidate` / `clear_all`）。
- `MemoryManager` 按类型路由，存储后端可依赖注入。
- **Working**：内存 + TTL + 容量 + TF-IDF/关键词混合检索。
- **Episodic**：PostgreSQL 同事务 outbox + Milvus；`vector_indexed_at` / `embedding_model`。
- **Semantic**：Neo4j 内 outbox + Milvus；`compute_graph_relevance` 统一图检索；RRF 向量/图分轨融合。
- **Perceptual**（experimental）：文本代理 embedding，默认关闭。
- Outbox 维护：stale 回收、死信重放、语义对账；`scripts/memory_vector_worker.py` / `memory_status.py`。
- **forget**：四类记忆，策略 `importance` / `importance_ttl` / `session`。
- **Embedding 迁移**：版本化 Milvus collection + 维度 guard + `scripts/memory_reindex.py`。
- Agent：`with_agent_tools(enable_memory=True)` 默认连带 memory hooks；`create_agent_tool_registry`。

## 当前妥协

- 概念抽取依赖 LLM（`metadata.concepts` 可覆盖）。
- 图扩展为启发式 hop1/hop2，RELATES_TO 权重可配置，写入时 weight 仍固定 1.0。
- Semantic / Episodic 向量索引为异步最终一致，需常驻 worker。
- `forget` 无 dry-run；semantic 批量删除需保守 threshold。
- Perceptual 非真实多模态向量空间。
- `MemoryTool` 默认仅 `working`，episodic/semantic 需显式 `memory_types`。

## 待办与问题台账

详见 [docs/system-issues.md](../docs/system-issues.md)。
