# 记忆系统：一致性问题与方案 backlog

本文档集中记录**数据一致性、可靠性、检索质量**相关的问题、当前实现、目标方案与优先级。  
实现状态摘要仍见 [implementation_status.md](./implementation_status.md)。

---

## 推荐方向：Postgres 事务性 Outbox + Worker（你提出的方案）

### 方案描述

1. **Postgres** 同时承载：
   - 业务表（如已有 `episodic_memories`；语义记忆若仍以 Neo4j 为主，可不重复存全文）
   - **`memory_vector_outbox`**：Milvus 待写入/待补偿任务
2. **`add` 路径**：在同一数据库事务内  
   `INSERT episodic_memories` + `INSERT memory_vector_outbox (status=pending)`  
   事务提交后，结构化数据与 outbox 条目**同时可见**。
3. **Worker**（定时或 LISTEN/NOTIFY）：
   - 查询 `pending`（可加 `FOR UPDATE SKIP LOCKED` 防并发重复处理）
   - 调用 embedding（若 outbox 只存 `content` 则 worker 再算向量；或 add 时已算好向量写入 outbox JSONB）
   - `upsert` Milvus
   - 成功 → `status=done`；失败 → `attempts++`、`last_error`、`next_retry_at`
4. **检索路径**：
   - 优先依赖 Milvus；可选在 `pending` 过多时对单条记录做**同步补偿**（读路径 flush，仅作降级）
   - 长期可给 `episodic_memories` 增加 `vector_indexed_at`，便于监控与 UI 提示

### 评价（结论：**值得做，且优于当前内存 outbox**）

| 维度 | 说明 |
|------|------|
| 正确性 | 标准 **Transactional Outbox**，比「先写 PG 再 try Milvus，失败仅进程内队列」可恢复 |
| 运维 | pending 可查、可告警、可重放；进程重启不丢任务 |
| 与现有代码 | `episodic` 天然贴合现有 `PostgresEpisodicMemoryStore`；改动面可控 |
| 需注意 | **Semantic** 主存是 Neo4j，不是 Postgres「双写同库事务」；outbox 仍可用同一 PG，但只能是 **Neo4j 提交后的 Milvus 补偿队列**，不能指望一个 PG 事务包住 Neo4j |
| 需补齐 | 幂等 upsert、多 worker 锁、死信队列、删除时的 **vector 删除 outbox**、可观测性（指标/日志） |

### 建议表结构（草案）

```sql
CREATE TABLE IF NOT EXISTS memory_vector_outbox (
    id BIGSERIAL PRIMARY KEY,
    memory_kind TEXT NOT NULL,          -- episodic | perceptual
    memory_id UUID NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT,
    collection_name TEXT NOT NULL,
    vector JSONB NOT NULL,              -- 或 content TEXT，由 worker 调 embed
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | done | dead
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 5,
    last_error TEXT,
    next_retry_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (memory_kind, memory_id)
);

CREATE INDEX idx_outbox_pending
    ON memory_vector_outbox (status, next_retry_at)
    WHERE status IN ('pending', 'processing');
```

### 与当前实现的对应关系

| 当前 | 目标 |
|------|------|
| `memory/storage/vector_outbox.py` 进程内 dict | `PostgresVectorOutboxStore` + 同上表 |
| `upsert_vector_with_outbox` 吞错入队 | PG 事务内入队；**无 outbox 配置时应显式失败**（避免静默丢向量） |
| `retrieve` / `search_memory` 前 `flush` | Worker 为主；读路径 flush 仅作可选降级 |
| 无 Worker | `scripts/memory_vector_worker.py` 或独立进程 / cron |

### 实施顺序建议

1. **Episodic**：PG 事务 outbox + worker（价值最大、模型最简单）
2. **Semantic**：Neo4j upsert 成功后写 PG outbox（独立事务，接受「极短窗口 Neo4j 有、Milvus 无」）
3. **Perceptual**：同上，按 modality collection 写入 outbox
4. 删除/更新：outbox 增加 `op` 字段（`upsert` / `delete`）
5. 弃用或仅测试环境保留内存 `VectorOutbox`

---

## 问题 backlog（集中台账）

状态：`open` | `mitigated` | `accepted` | `planned` | `done`

### 一致性 / 双写

| ID | 状态 | 问题 | 当前行为 | 目标/方案 | 优先级 |
|----|------|------|----------|-----------|--------|
| C-01 | done | Episodic/Semantic 写 Milvus 失败 | PG `memory_vector_outbox` + `memory_vector_worker.py` | 生产需常驻 worker；读路径可选 poll | P0 |
| C-02 | done | Outbox 非持久，重启丢任务 | `memory/storage/postgres_outbox_store.py` | 死信状态 `dead`；监控 pending 计数 | P0 |
| C-03 | done | 无 outbox 时 Milvus 失败静默 | `VectorWriteError` | episodic/semantic 在无 outbox 时抛错 | P0 |
| C-04 | mitigated | Semantic 无法与 Neo4j 同一 PG 事务 | **Neo4j 内 Transactional Outbox**：同事务写 `SemanticMemory` + `SemanticOutboxEvent`；Worker 按 `version` 同步 Milvus | 读路径 RYW 补 `embedding_status=pending`；Phase2 对账 | P1 |
| C-05 | done | Perceptual 无 outbox | PG `enqueue_upsert` + worker `perceptual` kind | 无 PG 时直接 upsert 或 `VectorWriteError` | P1 |
| C-06 | mitigated | `remove` 只删一端 | episodic/semantic/perceptual 可 `enqueue_delete` | Milvus 即时删失败仍靠 outbox | P1 |
| C-07 | done | `update` episodic/semantic 为删后重建 | **episodic / semantic / perceptual 均原地 update 保留 ID** | — | P2 |

### 检索质量（非双写，但影响「跑通」观感）

| ID | 状态 | 问题 | 当前行为 | 目标/方案 | 优先级 |
|----|------|------|----------|-----------|--------|
| R-01 | done | Working 中文子串搜不到 | 子串 + 字符 bigram 与 token 混合评分 | 可选 jieba / working embed | P0 |
| R-02 | done | Semantic 仅 Milvus 候选算图分 | `expand_graph_candidates` + `session_id` 过滤 | 合并重复 hop1 打分（R-03） | P1 |
| R-03 | open | 图扩展与 `score_related_memories` 重复 | 两次 Cypher/逻辑 | 合并为一次图检索 API | P2 |
| R-04 | open | 扩展候选 vector_score=0 | 仅靠 graph 分排前面 | 文档化策略或提高 graph 权重下限 | P2 |

### 「为跑通而兜底」（你明确反感的一类）

| ID | 状态 | 问题 | 当前行为 | 建议 | 优先级 |
|----|------|------|----------|------|--------|
| F-01 | done | LLM 概念抽取失败 | `concept_extraction_source` + 抛错 | LLM 失败直接 `RuntimeError` | P1 |
| F-02 | done | 未配 LLM 却开启抽取 | `logger.warning` 回退启发式 | 可选启动校验 | P2 |
| F-03 | done | 感知模态非法/缺 collection | 非法模态 `ValueError`；缺 collection 明确报错 | — | P1 |
| F-04 | done | 感知 recency 解析失败 | 非 ISO `timestamp` 抛 `ValueError` | — | P2 |
| F-05 | accepted | 图像/音频 embedding | caption/transcript 代理 | 产品标注为「文本代理」；远期 CLIP/CLAP | P3 |

### 架构 / 冗余 / 测试

| ID | 状态 | 问题 | 说明 | 建议 | 优先级 |
|----|------|------|------|------|--------|
| A-01 | done | outbox flush 双调用 | 仅 `MemoryManager.search_memory` 在 poll 时 flush | module `retrieve` 不再重复 flush | P2 |
| A-02 | open | `MemoryManagerProtocol` 重复 | tool 与 manager 双份接口 | Tool 直接依赖 `MemoryManager` 或共享 Protocol 单文件 | P3 |
| A-03 | done | 无 Neo4j 语义集成测试 | `tests/test_semantic_integration.py` | `RUN_SEMANTIC_INTEGRATION=1` + worker | P1 |
| A-04 | open | Agent 测 FakeMemoryManager | 未覆盖真实 MemoryTool 初始化 | 增加「最小真实 manager」或 testcontainer 测 | P2 |
| A-05 | open | `manager.py` 过重 | 工厂+CRUD+统计+清空 | 拆 `operations` / `factory` | P3 |
| A-06 | accepted | Working 以外不继承 BaseMemory | API 不统一 | 统一 Protocol 或文档说明 intentional | P3 |

---

## 设计原则（后续改代码时对齐）

1. **失败要可见**：能入队就入队并留下 `last_error`；不能入队就抛错，不要「假装写入成功」。
2. **补偿异步、状态可查**：Worker 为主；读路径 flush 不是唯一可靠手段。
3. **幂等**：Milvus upsert 按 `memory_id`；outbox 处理要容忍重复投递。
4. **分存储边界**：Episodic 用 PG 事务 outbox；Semantic 用 **Neo4j 内 outbox**（Milvus 可重建）；均最终一致。
5. **少静默兜底**：启发式、fallback 可以是显式配置路径，不是默认隐瞒。

---

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-06-03 | 初版：纳入 PG outbox + Worker 方案与审查台账 |
| 2026-06-03 | Semantic：**Neo4j 内 Outbox**（`SemanticOutboxEvent` + `SemanticOutboxProcessor` + RYW 读路径） |
