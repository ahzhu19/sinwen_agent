# 记忆系统架构

记忆系统位于 `memory/`，用四种记忆类型覆盖从秒级到天级的 Agent 上下文生命周期。

## 整体架构

```
MemoryTool （统一入口，9 个 action）
    │
    ▼
MemoryManager （按 memory_types 路由 + 依赖注入）
    │
    ├─► WorkingMemory     → InMemoryStore
    ├─► EpisodicMemory    → EpisodicMemoryStore (PostgreSQL) + MilvusVectorStore
    ├─► SemanticMemory    → Neo4jStore (Neo4j) + MilvusVectorStore
    └─► PerceptualMemory  → PerceptualMemoryStore (内存) + 多模态 MilvusVectorStore
```

存储后端全部通过 Protocol 接口暴露，测试时注入 Fake 实现。

## 四种记忆

### WorkingMemory — 会话记忆

- **存储**：内存 `InMemoryStore`
- **生命周期**：单次 Agent 会话，TTL 过期 + 容量上限自动淘汰
- **检索**：TF-IDF 向量化 + 关键词匹配 + 时间衰减 + 重要性加权
- **代码**：[`memory/modules/working.py`](../../memory/modules/working.py)

### EpisodicMemory — 情景记忆

- **存储**：PostgreSQL 存结构化事件，Milvus 存 embedding 向量
- **事务**：`episodic_memories` 与 `memory_vector_outbox` 在同一 PostgreSQL 事务中写入；memory-worker 异步消费 outbox 写入 Milvus
- **检索公式**：`(向量相似度 × 0.8 + 时间近因 × 0.2) × (0.8 + 重要性 × 0.4)`
- **Store Protocol**：[`memory/storage/postgres_store.py`](../../memory/storage/postgres_store.py)
- **Module**：[`memory/modules/episodic.py`](../../memory/modules/episodic.py)

### SemanticMemory — 语义记忆

- **存储**：Neo4j 存概念图（`Concept` 节点 + `MENTIONS` 关系），Milvus 存 embedding
- **事务**：Neo4j 同事务写 `SemanticMemory` + `SemanticOutboxEvent`；memory-worker 按 `version` embed 后写入 Milvus
- **概念抽取**：优先 `metadata.concepts`；无则调用 LLM（`prompts/memory.py` 中的 `SEMANTIC_CONCEPT_EXTRACTION_PROMPT`）写入 Neo4j
- **图扩展检索**：一跳直接概念匹配 + 二跳 `RELATES_TO` / 共现桥接，路径衰减系数 `SEMANTIC_GRAPH_HOP_DECAY`
- **检索公式**：`(向量相似度 × 0.7 + 图概念关系 × 0.3) × (0.8 + 重要性 × 0.4)`
- **Store**：[`memory/storage/neo4j_store.py`](../../memory/storage/neo4j_store.py)（930 行，包含图扩展和 outbox 事件）
- **Module**：[`memory/modules/semantic.py`](../../memory/modules/semantic.py)

### PerceptualMemory — 感知记忆

- **存储**：元数据落 `InMemoryPerceptualStore`，向量按模态路由到不同 Milvus collection
- **模态路由**：`text` / `image` / `audio` / `video` / `file` → 各自 collection
- **检索公式**：`(向量相似度 × 0.8 + 时间近因指数衰减 × 0.2) × (0.8 + 重要性 × 0.4)`
- **当前限制**：图像用 `caption` / `ocr_text` 代理，音频用 `transcript` 代理；跨模态检索是代理文本向量检索，非 CLIP/CLAP 统一向量空间
- **Module**：[`memory/modules/perceptual.py`](../../memory/modules/perceptual.py)

## Outbox / Saga 事务保证

双写（结构化 + 向量）的核心问题：先写一个再写另一个，第二个失败导致孤立数据。通过 outbox 模式求解：

| 记忆类型 | 主要存储 | Outbox 载体 | 处理方式 |
|----------|----------|-----------|---------|
| Episodic | PostgreSQL | `memory_vector_outbox`（同事务） | memory-worker 异步写 Milvus |
| Semantic | Neo4j | `SemanticOutboxEvent`（同事务） | memory-worker embed → Milvus |
| RAG | PostgreSQL | `rag_vector_outbox` | 同一 memory-worker 消费 |

Worker 入口：[`scripts/memory_vector_worker.py`](../../scripts/memory_vector_worker.py)

- `--loop --interval 10`：循环消费
- `--replay-dead`：重放失败事件
- `--no-reconcile-semantic`：跳过语义对账
- `--once --poll-timeout 30`：单次消费

查看积压状态：`uv run python scripts/memory_status.py`

详细设计见 [`memory/consistency_backlog.md`](../../memory/consistency_backlog.md)。

## MemoryTool — 统一入口

9 个 action，通过 `action` 参数路由：

| Action | 说明 | 支持的类型 |
|--------|------|-----------|
| `add` | 写入记忆 | working, episodic, semantic, perceptual |
| `search` | 检索记忆 | working, episodic, semantic, perceptual |
| `summary` | 记忆摘要 | working, episodic |
| `stats` | 数量统计 | working, episodic, semantic |
| `update` | 更新记忆 | working, episodic, semantic |
| `remove` | 删除记忆 | working, episodic, semantic |
| `forget` | 遗忘清理 | working（过期 + 低重要性） |
| `consolidate` | working → episodic 迁移 | 需同时启用两类 |
| `clear_all` | 批量清除 | 按用户 + 可选 session |

## 概念抽取器

[`memory/concept_extractor.py`](../../memory/concept_extractor.py)

- `LLMConceptExtractor`：调用 LLM（prompt 在 `prompts/memory.py`）从文本抽取概念词
- `StubConceptExtractor`：用于无 LLM 时的占位（返回空列表）
- 策略：`metadata.concepts` 优先 → LLM 抽取 → 写入 Neo4j `Concept` 节点及 `MENTIONS` 关系

## 存储后端

| 后端 | 文件 | 用途 |
|------|------|------|
| PostgreSQL | `storage/postgres_store.py` | Episodic 结构化存储 + outbox |
| Neo4j | `storage/neo4j_store.py` | Semantic 图存储 + 概念扩展 + outbox |
| Milvus | `storage/milvus_store.py` | 统一向量索引（episodic / semantic / perceptual / rag） |
| InMemoryStore | `storage/document_store.py` | Working 内存存储 + Perceptual 元数据 |
