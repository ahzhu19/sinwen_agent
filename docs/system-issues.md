# sinwen_agent 系统问题与待办清单

> 生成日期：2026-06-03  
> 用途：逐项讨论、排期、决策。每条带 **ID**，提问时可引用（如「C-04 怎么处理？」）。  
> 问题 ID 可引用（如「G-02 怎么处理？」）。  
> 模块实现摘要见 [memory/implementation_status.md](../memory/implementation_status.md)、[rag/implementation_status.md](../rag/implementation_status.md)。

---

## 如何使用本文档

1. 按 **状态** 筛选：`open` / `mitigated` / `accepted` / `done`。
2. 按 **优先级** 排序：P0 > P1 > P2 > P3。

**测试基线**：`uv run pytest tests/ -q` → 276 passed, 3 skipped。

---

## 一、已解决（近期，供对照）

| ID | 标题 | 摘要 |
|----|------|------|
| C-01 | Milvus 双写失败补偿 | PG `memory_vector_outbox` + `scripts/memory_vector_worker.py` |
| C-02 | Outbox 持久化 | `postgres_outbox_store.py`，支持 `dead` 状态 |
| C-03 | 无 outbox 时静默失败 | 抛 `VectorWriteError` |
| C-05 | Perceptual outbox | PG enqueue + worker `perceptual` |
| C-07 | update 删后重建 | episodic / semantic / perceptual 原地 update 保留 ID |
| R-01 | Working 中文检索 | 子串 + bigram |
| R-02 | 语义图仅算 Milvus 候选 | `expand_graph_candidates` + session 过滤 |
| F-01~F-04 | 静默兜底类 | 非法模态、坏 timestamp、LLM 失败抛错等 |
| A-01 | outbox flush 重复 | 仅 `MemoryManager.search_memory` poll 时 flush |
| A-03 | 语义真机测试 | `RUN_SEMANTIC_INTEGRATION=1` |

---

## 二、记忆系统 — 一致性 / 双写（open & mitigated）

### C-04 · Semantic 与 Neo4j 无法 PG 同事务

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P1 |
| **问题** | 语义记忆主存是 Neo4j，Milvus 向量索引仍是异步最终一致。 |
| **当前行为** | Neo4j 同事务写 `SemanticMemory` + `SemanticOutboxEvent`；Worker 按 `version` 同步 Milvus；读路径补 `embedding_status=pending`。Worker 启动时 `reclaim_stale_processing_outbox`；`ensure_pending_outbox_events` 为缺 outbox 的记忆补 pending 事件。 |
| **目标方案** | ~~Phase 2 增加 Neo4j ↔ Milvus 对账、processing 超时回收~~（已实现）；换 embedding 模型后的重建策略已实现：`scripts/memory_reindex.py` 覆盖 episodic / semantic / RAG 三类，支持 `--dry-run` / `--rag-only` / `--episodic-only` / `--semantic-only`。 |
| **相关代码** | `memory/modules/semantic.py`、`memory/storage/neo4j_store.py`、`memory/semantic_outbox_processor.py` |
| **可讨论的问题** | 对账频率如何定？是否需要 UI/CLI 展示「索引中」？ |

---

### C-06 · remove 双端删除不完全可靠

| 字段 | 内容 |
|------|------|
| **状态** | `mitigated`（死信重放已实现） |
| **优先级** | P1 |
| **问题** | 删除时 PG/Neo4j 与 Milvus 可能不同步。 |
| **当前行为** | 支持 `enqueue_delete`；Milvus **即时** delete 失败时依赖 outbox 补偿。PG / Neo4j outbox 均支持 `replay_dead()`；Worker CLI 提供 `--replay-dead`。 |
| **目标方案** | ~~监控 `dead` 队列~~（`memory_status.py` 已展示）；必要时读路径校验孤儿向量。 |
| **相关代码** | `memory/modules/episodic.py`、`semantic.py`、`perceptual.py` |
| **可讨论的问题** | 是否需要「删除确认」API？死信如何处理？ |

---

### OPS-01 · 生产必须常驻 Vector Worker

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P0 |
| **问题** | 有 `DATABASE_URL` 时，Milvus 写入依赖 Worker，不能只靠读路径 flush。 |
| **当前行为** | `docker-compose.yml` 已包含 `memory-worker`，也可手动运行 `uv run python scripts/memory_vector_worker.py --loop`。 |
| **操作** | `docker compose up` 或 `uv run python scripts/memory_vector_worker.py --loop` |
| **可讨论的问题** | 生产环境使用 compose、systemd 还是 k8s？ |

---

### OPS-02 · 缺少 vector_indexed_at / 可观测性

| 字段 | 内容 |
|------|------|
| **状态** | `mitigated`（episodic 字段 + CLI 已加） |
| **优先级** | P1 |
| **问题** | 业务表无「已向量化」字段；已有 outbox `pending` / `processing` / `dead` 计数，但尚无 Prometheus 告警。 |
| **当前行为** | `episodic_memories.vector_indexed_at`；Worker upsert 成功后 `mark_vector_indexed`；`scripts/memory_status.py` 汇总 Postgres / Neo4j / RAG outbox 及未索引计数；Worker 每轮先跑 `run_memory_outbox_maintenance`。 |
| **目标方案** | Prometheus 指标或告警规则。 |
| **可讨论的问题** | 需要哪些指标？UI 要不要展示「索引中」？ |

---

### OPS-03 · 无 DATABASE_URL 时内存 outbox

| 字段 | 内容 |
|------|------|
| **状态** | `accepted` |
| **优先级** | P2 |
| **问题** | 单测/本地无 PG 时用进程内 `VectorOutbox`，**进程重启丢任务**。 |
| **当前行为** | 适合测试；不适合生产。 |
| **可讨论的问题** | 是否要在启动时 warn？ |

---

## 三、记忆系统 — 检索质量（open）

### R-03 · 图扩展与 score_related_memories 重复

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **当前行为** | `compute_graph_relevance` 统一 hop1/hop2；`expand_graph_candidates` / `score_related_memories` 为薄包装。 |

---

### R-04 · 图扩展候选 vector_score=0

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **当前行为** | 向量榜 + 图榜 **RRF** 融合（`semantic_retrieve.py`，`SEMANTIC_RRF_K`），再乘 importance。 |

---

### R-05 · 图关系类型无权重配置

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **当前行为** | `SEMANTIC_GRAPH_RELATION_WEIGHTS` 配置 RELATES_TO / CO_OCCURRENCE；写入时 `RELATES_TO` 关系动态递增 `co_occurrence_count`，读取时 `compute_graph_relevance` 用 PMI-inspired log boost 调整 hop2 分数。 |
| **解决方案** | Neo4j Cypher `MERGE ... ON CREATE SET co_occurrence_count=1, weight=1.0, ON MATCH SET co_occurrence_count = co_occurrence_count + 1`；hop2 查询增加 `avg(co_occurrence_count)`，分数计算用 `pmi_boost = min(2.0, 1.0 + log(co_occurrence) / 2)`。 |
| **剩余** | 真机 Neo4j 验证 PMI 数值合理性。 |

---

### R-06 · Working / Semantic 无正式中文分词

| 字段 | 内容 |
|------|------|
| **状态** | `done`（Working 已接入 jieba） |
| **优先级** | P3 |
| **当前行为** | `memory/tokenizer.py` 提供 `tokenize()` 函数，优先使用 jieba 做中文词级别分词，不可用时降级为正则。WorkingMemory.`_tokenize` 已替换为调用此函数。TF-IDF 和关键词匹配从逐字切分升级为词级别（如"机器学习"→\["机器", "学习"\]）。 |
| **相关代码** | `memory/tokenizer.py`、`memory/modules/working.py` |
| **测试** | `tests/test_chinese_tokenizer.py`（6 测试） |
| **剩余** | Semantic 侧仍依赖 LLM 概念抽取，非关键词分词。 |

---

## 四、记忆系统 — 产品能力缺口（open）

### M-01 · forget 策略

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **当前行为** | 四类记忆支持 `forget`；策略 `importance` / `importance_ttl` / `session`；semantic 默认单次上限 50 条。`dry_run=True` 返回预览列表（`list[MemoryRecord]`）不执行删除，Manager / Service / Tool 三层透传。 |
| **剩余** | 更保守的 semantic 删除确认。 |

---

### M-02 · MemoryTool 默认仅 working

| 字段 | 内容 |
|------|------|
| **状态** | `accepted` |
| **优先级** | P2 |
| **问题** | 避免默认初始化强依赖 PG/Neo4j/Milvus。 |
| **当前行为** | 需 `memory_types=["working","episodic",...]` 显式传入。 |
| **可讨论的问题** | 是否提供「全开」预设？环境变量默认列表？ |

---

### M-04 · 更换 embedding 模型导致 Milvus 维度不匹配

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **当前行为** | `USE_VERSIONED_MILVUS_COLLECTIONS` 版本化 collection；Milvus 维度 guard；`scripts/memory_reindex.py`；episodic `embedding_model` 列。episodic / semantic / perceptual 均已接入 `resolve_collection_name`。 |
| **解决方案** | RAG 侧补齐版本化：`RagConfig` 增加 `use_versioned_milvus_collections` / `embed_model_name` / `rag_milvus_collection()` 方法；`RagManager` 改用 `rag_milvus_collection()`；`MilvusRagVectorStore.ensure_collection` 增加 `validate_collection_dimension` guard。`from_env` 解析 `USE_VERSIONED_MILVUS_COLLECTIONS` / `EMBED_MODEL_NAME`。 |
| **测试** | `tests/test_rag_collection_versioning.py`（6 测试） |

---

### M-05 · metadata 浅拷贝与副作用

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P3 |
| **当前行为** | 所有 `dict(metadata)` 入口替换为 `copy.deepcopy(metadata)`：`BaseMemory.add`、`WorkingMemory.add`、`InMemoryStore.update`、`MemoryManager.update_memory`、`MemoryTool._add_memory/_update_memory`、episodic/semantic/perceptual 模块。嵌套 dict/list 不再与调用方共享引用。 |
| **测试** | `tests/test_metadata_deepcopy.py`（3 测试） |

---

### M-06 · InMemoryStore 未全局统一

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P3 |
| **当前行为** | `BaseMemory.store` 重命名为 `_store`（私有），与 Episodic/Semantic/Perceptual 模块一致。`BaseMemory` 提供 `store` property 保持向后兼容。WorkingMemory 内部全部改用 `self._store`。MemoryManager 的 `self.store` 同步改为 `self._store`。 |

---

## 五、记忆系统 — 架构与测试（open & accepted）

### A-02 · MemoryManagerProtocol 重复

| 字段 | 内容 |
|------|------|
| **状态** | `mitigated` |
| **优先级** | P3 |
| **当前行为** | `MemoryServiceProtocol` + `MemoryTool` 适配器；`memory_manager=` 构造参数保留兼容。 |

---

### A-04 · Agent 测试多用 Fake，少覆盖真实初始化

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **当前行为** | `tests/test_real_memory_init.py` 覆盖真实初始化路径：MemoryConfig → MemoryManager → MemoryService → MemoryTool → ToolRegistry → SimpleAgent / ReActAgent，仅用 working memory 的 InMemoryStore（无需 PG/Milvus/Neo4j）。6 个测试覆盖 add/search/stats/forget(dry_run+execute) 及 Agent 工具调用全链路。 |
| **测试** | `tests/test_real_memory_init.py`（6 测试） |

---

### A-05 · manager.py 过重

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P3 |
| **当前行为** | `memory/manager.py` 从 610 行降至 120 行。工厂逻辑抽取到 `memory/factory.py`（`setup_outbox` / `create_episodic_memory` / `create_semantic_memory` / `create_perceptual_memory`）；CRUD / forget / consolidate / clear / stats / outbox 操作抽取到 `memory/operations.py`（`MemoryOperations` mixin）。`MemoryManager` 继承 mixin 并委托工厂函数，对外 API 完全不变。 |

---

### A-07 · Agent Runtime Memory Hooks

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **当前行为** | `MemoryHookConfig` 默认检索 working+episodic；`record_interaction` 单条合并 Q/A；`enable_memory=True` 默认开启 hooks。ReflectionAgent 反思后写入 semantic（AG-01 done）；PlanAndSolveAgent 规划前检索 episodic、完成后存入 episodic（AG-02 done）。 |

---

### A-06 · Working 外不继承 BaseMemory

| 字段 | 内容 |
|------|------|
| **状态** | `accepted` |
| **优先级** | P3 |
| **问题** | Episodic/Semantic/Perceptual API 形状相近但未统一基类。 |
| **可讨论的问题** | 统一 Protocol 还是保持 intentional 分叉？ |

---

## 六、感知记忆 — 已知妥协（accepted）

### F-05 / P-01 · 文本代理 embedding（非 CLIP/CLAP）

| 字段 | 内容 |
|------|------|
| **状态** | `accepted` |
| **优先级** | P3 |
| **问题** | 图像用 caption/ocr，音频用 transcript，非统一多模态向量空间。 |
| **当前行为** | 按模态分 Milvus collection；跨模态检索为代理文本向量。 |
| **可讨论的问题** | 何时接入真实模型？跨模态分数如何归一化？ |

---

## 七、RAG 子系统（open）

### G-01 · 摄取 PG + Milvus 非事务

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P1 |
| **问题** | 与 memory 早期相同：元数据写入 PG 后 Milvus 失败可能孤立。 |
| **当前行为** | `rag_vector_outbox` 表 + `RagIngestionService` 入队；`scripts/memory_vector_worker.py` 统一消费 memory + RAG outbox。 |
| **相关文档** | [rag/implementation_status.md](../rag/implementation_status.md) |
| **目标方案** | ~~复用 memory outbox 或独立补偿任务~~（已实现独立 RAG outbox，同一 Worker 处理）。 |

---

### G-02 · 无 reranker

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **当前行为** | `rag/reranker.py` 提供 `Reranker` Protocol + `NoneReranker`（透传）+ `LLMReranker`（LLM 打分重排序，失败回退向量分数）；`create_reranker` 工厂。`retriever.search` 启用 `rerank` 时候选扩大 `rerank_candidate_factor×top_k`（默认 3x），rerank 后截断 top_k。`RagManager` → `RagTool` 透传 `rerank` 参数。 |
| **相关代码** | `rag/reranker.py`、`rag/retriever.py`、`rag/manager.py`、`tools/builtin/rag_tool.py` |
| **测试** | `tests/test_rag_reranker.py`（14 测试） |

---

### G-03 · URL / 目录批量摄取

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P3 |
| **问题** | MVP 仅本地单文件。 |
| **解决方案** | RagManager 增加 `ingest_url`（下载到临时文件，source_uri 保留原始 URL）和 `ingest_directory`（glob 遍历，返回 BatchIngestResult 含成功/失败列表）。RagTool source_type 支持 file / url / directory。 |

---

### G-04 · HyDE / multi_query 依赖额外 LLM

| 字段 | 内容 |
|------|------|
| **状态** | `accepted` |
| **优先级** | P2 |
| **问题** | 真机需 LLM API；成本与延迟需知情。 |

---

## 八、Agent 与跨模块集成（open）

### AG-01 · ReflectionAgent 语义记忆沉淀

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **问题** | 自我批评结果未自动写入 semantic。 |
| **解决方案** | ReflectionAgent 增加 `memory_service` / `enable_memory` 参数。反思循环结束后，`_maybe_record_reflection` 将问题、批评摘要、最终答案写入 semantic 记忆（importance=0.7）。写入失败不影响 Agent 正常返回。`with_agent_tools(enable_memory=True)` 自动创建 MemoryService。向后兼容：默认不启用。 |
| **测试** | `tests/test_reflection_memory.py`（6 测试） |

---

### AG-02 · PlanAndSolveAgent 情景记忆复用

| 字段 | 内容 |
|------|------|
| **状态** | `done` |
| **优先级** | P2 |
| **问题** | 历史计划未从 episodic 检索复用。 |
| **解决方案** | PlanAndSolveAgent 增加 `memory_service` / `enable_memory` 参数。规划前 `_retrieve_past_plans` 从 episodic 检索相似历史计划并注入 planner 上下文；完成后 `_maybe_record_plan` 将问题、计划、最终答案存入 episodic 记忆（importance=0.6）。检索/写入失败均不影响 Agent 正常运行。`with_agent_tools(enable_memory=True)` 自动创建 MemoryService。向后兼容：默认不启用。 |
| **测试** | `tests/test_plan_solve_memory.py`（8 测试） |

---

### AG-03 · Reflection / PlanAndSolve 未默认 memory & rag

| 字段 | 内容 |
|------|------|
| **状态** | `open`（**用户曾明确：暂不做默认闭环**） |
| **优先级** | P3 |
| **当前行为** | 可用 `create_agent_tool_registry` 手动注册。 |
| **真机脚本** | `scripts/try_memory_agent.py` |

---

### AG-04 · SimpleAgent / ReAct 记忆默认关闭

| 字段 | 内容 |
|------|------|
| **状态** | `accepted` |
| **优先级** | P3 |
| **问题** | 需 `enable_memory=True` 才注册 MemoryTool。 |
| **可讨论的问题** | 默认开启会不会拖慢无 DB 场景？ |

---

## 九、概念抽取（当前策略说明）

### SEM-01 · 概念抽取始终走 LLM

| 字段 | 内容 |
|------|------|
| **状态** | `done`（策略变更） |
| **说明** | `metadata.concepts` 可覆盖；否则调用 LLM（`prompts/memory.py`），**无正则启发式兜底**。 |
| **配置** | `LLM_MODEL_ID` / `LLM_API_KEY`；`LLM_BASE_URL` 未设时回退 `EMBED_BASE_URL`。 |
| **失败** | 抛错（见 `memory/concept_extractor.py`），写入 `concept_extraction_source` 等 metadata。 |
| **可讨论的问题** | 无 LLM 时是否允许仅 metadata.concepts？离线测试怎么办？ |

---

## 十、设计原则（改代码时对齐）

1. **失败要可见**：能入队就入队并留 `last_error`；不能入队就抛错。
2. **补偿异步、状态可查**：Worker 为主；读路径 flush 仅降级。
3. **幂等**：Milvus upsert 按 `memory_id`；outbox 容忍重复投递。
4. **分存储边界**：Episodic 可 PG 事务 outbox；Semantic 接受 Neo4j + PG outbox **最终一致**。
5. **少静默兜底**：fallback 须显式配置，不默认隐瞒。

---

## 十一、建议讨论顺序（可选）

1. **G-02 / A-04** — RAG reranker 与 Agent 真机集成测试  
2. **OPS-02** — Prometheus 告警  
3. **AG-01~02** — Agent 记忆沉淀（产品排期）  
4. **F-05** — 多模态远期  

---

## 十二、相关文档索引

| 文档 | 路径 |
|------|------|
| 记忆架构 | [docs/architecture/memory.md](./architecture/memory.md) |
| 记忆实现状态 | [memory/implementation_status.md](../memory/implementation_status.md) |
| RAG 实现状态 | [rag/implementation_status.md](../rag/implementation_status.md) |
| 项目 README | [README.md](../README.md) |
| 提示词 | [prompts/](../prompts/) |

---

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-06-03 | 初版：从对话整理，供逐项提问 |
