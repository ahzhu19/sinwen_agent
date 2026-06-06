# 记忆系统实现状态

## 已实现

- `MemoryTool.execute()` 统一入口，支持按 action 分发。
- `MemoryManager` 按启用类型初始化并分发到对应记忆模块。
- `WorkingMemory`：内存存储、TTL 清理、容量控制、会话检索、混合关键词/向量式评分。
- `EpisodicMemory`：PostgreSQL 结构化事件存储 + Milvus 向量检索的第一版编排。
- `SemanticMemory`：Neo4j 图存储 + Milvus 向量检索；**Neo4j 内 Transactional Outbox**（`version` / `embedding_status` / Worker 同步 Milvus）。
- `PerceptualMemory`（**experimental**）：元数据仅进程内存；图像/音频使用 caption/transcript 文本代理 embedding；默认 `enable_perceptual=False`。
- **产品化 Tool action**：`add` / `search` / `summary` / `stats` / `update` / `remove` / `forget` / `consolidate` / `clear_all`。
- `search` 已支持 `working`（通过 `WorkingMemory.retrieve`）。
- Agent 集成：`create_agent_tool_registry(enable_memory=True)`、`SimpleAgent` / `ReActAgent.with_agent_tools`、系统提示词。

## 当前妥协

- 概念抽取始终走 LLM（`metadata.concepts` 可覆盖）；需配置 `LLM_*` 或 `EMBED_BASE_URL` 回退。
- 图扩展检索为启发式邻居（RELATES_TO + 共现桥接），非全量图遍历。
- 默认启用 Postgres `memory_vector_outbox`（episodic/perceptual）；语义记忆向量经 **Neo4j `SemanticOutboxEvent`** 异步同步（需 `NEO4J_PASSWORD`）。
- 需运行 `scripts/memory_vector_worker.py` 异步写 Milvus（含 Neo4j semantic outbox）；`VECTOR_OUTBOX_POLL_ON_READ=true` 时检索前会同步补偿一批。
- `update`：episodic / semantic / perceptual 均支持原地更新并保留 ID。
- `forget` 仅实现 working 策略；`clear_all` 对 semantic 按 user/session 批量删除。
- `MemoryTool` 默认只启用 `working`；episodic/semantic 需显式传入 `memory_types`。
- PerceptualMemory（experimental）仍使用文本代理 embedding，非真实多模态模型；生产环境优先 RAG 或 semantic。

## 一致性与技术债台账

详见 [consistency_backlog.md](./consistency_backlog.md)（Postgres `memory_vector_outbox` + Worker 方案、问题 ID、优先级）。

## 后续建议

- 图扩展：关系类型权重配置（session 过滤已实现）。
- Perceptual 真实多模态 embedding 与跨模态归一化。
- 可选：`RUN_SEMANTIC_INTEGRATION=1` 真机回归 Neo4j + Milvus + LLM 概念抽取。
