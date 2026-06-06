# sinwen_agent 系统问题与待办清单

> 生成日期：2026-06-03  
> 用途：逐项讨论、排期、决策。每条带 **ID**，提问时可引用（如「C-04 怎么处理？」）。  
> 详细记忆台账见 [memory/consistency_backlog.md](../memory/consistency_backlog.md)。

---

## 如何使用本文档

1. 按 **状态** 筛选：`open`（未解决）、`mitigated`（部分缓解）、`accepted`（已知妥协）、`done`（已解决）。
2. 按 **优先级** 排序：P0 > P1 > P2 > P3。
3. 讨论某条时，复制其 **ID** 与「可讨论的问题」小节。

**测试基线**（最近一次）：`uv run pytest tests/ -q` → 156 passed, 3 skipped。  
**工作区**：记忆系统大改 diff 尚未提交 git。

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
| **状态** | `mitigated`（Phase 2 已实现） |
| **优先级** | P1 |
| **问题** | 语义记忆主存是 Neo4j，Milvus 向量索引仍是异步最终一致。 |
| **当前行为** | Neo4j 同事务写 `SemanticMemory` + `SemanticOutboxEvent`；Worker 按 `version` 同步 Milvus；读路径补 `embedding_status=pending`。Worker 启动时 `reclaim_stale_processing_outbox`；`ensure_pending_outbox_events` 为缺 outbox 的记忆补 pending 事件。 |
| **目标方案** | ~~Phase 2 增加 Neo4j ↔ Milvus 对账、processing 超时回收~~（已实现）；换 embedding 模型后的重建策略仍待做。 |
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
| **状态** | `open` |
| **优先级** | P2 |
| **问题** | 语义检索中图相关逻辑可能执行两次 Cypher/打分路径。 |
| **当前行为** | `expand_graph_candidates` 与 `score_related_memories` 并存。 |
| **目标方案** | 合并为一次图检索 API，减少延迟与重复。 |
| **相关代码** | `memory/storage/neo4j_store.py`、`memory/modules/semantic.py` |
| **可讨论的问题** | 合并后评分公式是否变化？要不要 benchmark？ |

---

### R-04 · 图扩展候选 vector_score=0

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P2 |
| **问题** | 仅从图扩展进来的记忆没有 Milvus 相似度，排序主要靠 graph 分。 |
| **当前行为** | 公式 `(vector * 0.7 + graph * 0.3) * importance_weight`，扩展候选 vector 项为 0。 |
| **目标方案** | 文档化策略；或提高 graph 权重下限 / 对纯图候选单独排序。 |
| **可讨论的问题** | 产品上「纯概念关联」排前面是否合理？ |

---

### R-05 · 图关系类型无权重配置

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P2 |
| **问题** | RELATES_TO 与共现桥接均为启发式，关系类型权重不可配置。 |
| **当前行为** | `SEMANTIC_GRAPH_HOP_DECAY`、`SEMANTIC_GRAPH_MAX_HOPS` 可配。 |
| **目标方案** | 按关系类型配置权重（implementation_status 已列）。 |
| **可讨论的问题** | 需要哪些关系类型？默认值从哪来？ |

---

### R-06 · Working / Semantic 无正式中文分词

| 字段 | 内容 |
|------|------|
| **状态** | `open`（R-01 已缓解 Working） |
| **优先级** | P3 |
| **问题** | 无 jieba 等分词；语义侧依赖 LLM 概念而非关键词。 |
| **可讨论的问题** | 是否引入 jieba？Working 是否要上 embedding？ |

---

## 四、记忆系统 — 产品能力缺口（open）

### M-01 · forget 仅支持 working

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P2 |
| **问题** | `forget` 只做 working 过期 + 低重要性删除；episodic/semantic/perceptual 无策略。 |
| **相关代码** | `memory/manager.py` → `forget_memories` |
| **可讨论的问题** | 其他类型需要哪些 forget 语义（按时间？按 session？）？ |

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

### M-03 · auto_classify 未实现

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P3 |
| **问题** | `add_memory(..., auto_classify=True)` 为 TODO，当前忽略。 |
| **相关代码** | `memory/manager.py` |
| **可讨论的问题** | 实现分类器还是删除参数？ |

---

### M-04 · 更换 embedding 模型导致 Milvus 维度不匹配

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P2 |
| **问题** | 换 `EMBED_MODEL_NAME` 后旧 collection 维度不一致。 |
| **可讨论的问题** | 是否需要 reindex 脚本 / 版本化 collection 名？ |

---

### M-05 · metadata 浅拷贝与副作用

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P3 |
| **问题** | README 曾提 `_add_memory` 修改调用方 dict；当前 Tool 已 `dict(metadata)`，但嵌套对象仍可能共享。 |
| **可讨论的问题** | 是否统一 `copy.deepcopy`？ |

---

### M-06 · InMemoryStore 未全局统一

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P3 |
| **问题** | WorkingMemory 仍保留旧模块引用，与 `InMemoryStore` 按类型索引未完全统一。 |
| **可讨论的问题** | 重构范围多大？ |

---

## 五、记忆系统 — 架构与测试（open & accepted）

### A-02 · MemoryManagerProtocol 重复

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P3 |
| **问题** | Tool 与 Manager 各有一份 Protocol 定义。 |
| **目标方案** | 单文件共享或 Tool 直接依赖 `MemoryManager`。 |

---

### A-04 · Agent 测试多用 Fake，少覆盖真实初始化

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P2 |
| **问题** | `test_simple_agent_memory` 等未走真实 `MemoryTool` + DB 配置路径。 |
| **目标方案** | 最小真实 manager 或 testcontainer。 |
| **可讨论的问题** | CI 是否加 Docker job？ |

---

### A-05 · manager.py 过重

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P3 |
| **问题** | 工厂 + CRUD + 统计 + 清空 + outbox 在同一文件（~500 行）。 |
| **目标方案** | 拆 `operations` / `factory`。 |

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
| **状态** | `open` |
| **优先级** | P2 |
| **问题** | `direct` / `hyde` / `multi_query` 按 chunk_id 取最高分合并，无 CrossEncoder。 |
| **可讨论的问题** | 先上 rerank 还是 HyDE 调参？ |

---

### G-03 · URL / 目录批量摄取未做

| 字段 | 内容 |
|------|------|
| **状态** | `open` |
| **优先级** | P3 |
| **问题** | MVP 仅本地单文件；设计文档已 defer。 |
| **相关文档** | [docs/superpowers/specs/2026-06-03-rag-system-design.md](./superpowers/specs/2026-06-03-rag-system-design.md) |

---

### G-04 · HyDE / multi_query 依赖额外 LLM

| 字段 | 内容 |
|------|------|
| **状态** | `accepted` |
| **优先级** | P2 |
| **问题** | 真机需 LLM API；成本与延迟需知情。 |

---

## 八、Agent 与跨模块集成（open）

### AG-01 · ReflectionAgent 未接语义记忆沉淀

| 字段 | 内容 |
|------|------|
| **状态** | `open`（产品排期） |
| **优先级** | P2 |
| **问题** | 自我批评结果未自动写入 semantic。 |
| **可讨论的问题** | 写入时机？由 Agent 调 Tool 还是框架自动？ |

---

### AG-02 · PlanAndSolveAgent 未接情景记忆复用

| 字段 | 内容 |
|------|------|
| **状态** | `open`（产品排期） |
| **优先级** | P2 |
| **问题** | 历史计划未从 episodic 检索复用。 |

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

1. **OPS-01 / C-04** — 生产部署与 Semantic 一致性是否可接受  
2. **G-01** — RAG 是否复用 memory outbox  
3. **R-03 / R-04** — 语义检索质量与排序策略  
4. **A-04** — 集成测试与 CI  
5. **AG-01~03** — Agent 产品闭环（按你的排期）  
6. **F-05 / P-01** — 多模态远期  

---

## 十二、相关文档索引

| 文档 | 路径 |
|------|------|
| 记忆一致性 backlog | [memory/consistency_backlog.md](../memory/consistency_backlog.md) |
| 记忆实现状态 | [memory/implementation_status.md](../memory/implementation_status.md) |
| RAG 实现状态 | [rag/implementation_status.md](../rag/implementation_status.md) |
| 项目 README 未完成节 | [README.md](../README.md) |
| 提示词目录 | [prompts/](../prompts/) |
| RAG 设计 spec | [docs/superpowers/specs/2026-06-03-rag-system-design.md](./superpowers/specs/2026-06-03-rag-system-design.md) |

---

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-06-03 | 初版：从对话整理，供逐项提问 |
