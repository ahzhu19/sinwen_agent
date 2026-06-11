# 记忆系统架构

记忆系统位于 `memory/`，用四种记忆类型覆盖从秒级到天级的 Agent 上下文生命周期。

## 整体架构

```
Agent Runtime / Memory Hooks
    │
    ▼
MemoryService（内部稳定 API）
    │
    ├─► MemoryTool（LLM/用户显式操作的 Tool Adapter）
    │
    ▼
MemoryManager （按 memory_types 路由 + 依赖注入）
    │
    ├─► WorkingMemory     → InMemoryStore
    ├─► EpisodicMemory    → PostgreSQL + MilvusVectorStore
    ├─► SemanticMemory    → Neo4j + MilvusVectorStore
    └─► PerceptualMemory  → 内存元数据 + 多模态 MilvusVectorStore
```

`MemoryService` 是 Agent 内部使用的记忆边界；`MemoryTool` 只是该服务的 Tool Adapter。

**Runtime Hooks**（`memory/hooks.py` + `agents/memory_runtime.py`）：
- run 前：`retrieve_context` 检索相关记忆并注入 system prompt（默认 **working + episodic**）
- run 后：`record_interaction` 将整轮对话合并为一条 working 记录
- `enable_memory=True` 时默认连带开启 hooks；可用 `enable_memory_hooks=False` 关闭
- hooks 可与 MemoryTool 独立（仅 hooks 不必注册 memory 工具）

存储后端通过 Protocol 暴露，测试时注入 Fake 实现。

## 四种记忆

### WorkingMemory

- **存储**：内存 `InMemoryStore`
- **生命周期**：TTL + 容量淘汰
- **检索**：TF-IDF + 关键词 + 时间衰减 + 重要性
- **代码**：[`memory/modules/working.py`](../../memory/modules/working.py)

### EpisodicMemory

- **存储**：PostgreSQL 事件 + Milvus 向量
- **事务**：`episodic_memories` 与 `memory_vector_outbox` 同事务；worker 写 Milvus 后更新 `vector_indexed_at`
- **检索**：`(向量 × 0.8 + 时间近因 × 0.2) × importance_weight`
- **代码**：[`memory/modules/episodic.py`](../../memory/modules/episodic.py)

### SemanticMemory

- **存储**：Neo4j 概念图 + Milvus 向量
- **事务**：Neo4j 同事务写 `SemanticMemory` + `SemanticOutboxEvent`
- **概念抽取**：`metadata.concepts` 优先，否则 LLM（[`prompts/memory.py`](../../prompts/memory.py)）
- **图检索**：`compute_graph_relevance`（hop1 概念匹配 + hop2 RELATES_TO / 共现桥接）
- **排序**：向量榜与图榜 **RRF 融合**（`SEMANTIC_RRF_K`），再乘 importance；关系权重 `SEMANTIC_GRAPH_RELATION_WEIGHTS`
- **代码**：[`memory/modules/semantic.py`](../../memory/modules/semantic.py)、[`memory/modules/semantic_retrieve.py`](../../memory/modules/semantic_retrieve.py)

### PerceptualMemory（experimental）

- 元数据内存 + 按模态 Milvus collection
- 图像/音频当前为 caption/transcript 文本代理向量

## Outbox 与 Worker

| 类型 | Outbox | Worker |
|------|--------|--------|
| Episodic / Perceptual | PG `memory_vector_outbox` | 异步 Milvus upsert/delete |
| Semantic | Neo4j `SemanticOutboxEvent` | embed + Milvus |
| RAG | PG `rag_vector_outbox` | 同一 worker |

入口：[`scripts/memory_vector_worker.py`](../../scripts/memory_vector_worker.py)（`--loop`、`--replay-dead`）  
状态：[`scripts/memory_status.py`](../../scripts/memory_status.py)  
换 embedding：[`scripts/memory_reindex.py`](../../scripts/memory_reindex.py)

Milvus collection 默认版本化：`USE_VERSIONED_MILVUS_COLLECTIONS=true` → `{base}_{model}_{dim}`。

## MemoryTool action 摘要

| Action | 支持类型 | 说明 |
|--------|----------|------|
| add / search / update / remove | 四类 | 基本 CRUD |
| forget | 四类 | `importance` / `importance_ttl` / `session` |
| consolidate | working→episodic | 需同时启用 |
| clear_all | 按类型批量 | semantic 支持 user/session |

## 问题台账

开放项与优先级见 [docs/system-issues.md](../system-issues.md)。  
模块妥协摘要见 [memory/implementation_status.md](../../memory/implementation_status.md)。
