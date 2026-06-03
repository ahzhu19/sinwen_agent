# 记忆系统实现状态

## 已实现

- `MemoryTool.execute()` 统一入口，支持按 action 分发。
- `MemoryManager` 按启用类型初始化并分发到对应记忆模块。
- `WorkingMemory`：内存存储、TTL 清理、容量控制、会话检索、混合关键词/向量式评分。
- `EpisodicMemory`：PostgreSQL 结构化事件存储 + Milvus 向量检索的第一版编排。
- `SemanticMemory`：Neo4j 图存储 + Milvus 向量检索的第一版编排。
  - 写入时同步写 Neo4j 语义图与 Milvus 向量库。
  - 检索公式为 `(向量相似度 * 0.7 + 图相似度 * 0.3) * (0.8 + 重要性 * 0.4)`。
  - Neo4j 中保存 `SemanticMemory`、`Concept`，并用 `MENTIONS` 关系连接。
- `PerceptualMemory`：感知记忆第一版编排。
  - 按 `text/image/audio/video/file` 模态路由到不同 Milvus collection。
  - 元数据写入 `InMemoryPerceptualStore`，原始数据只保存路径或 URI。
  - 检索公式为 `(向量相似度 * 0.8 + 时间近因性 * 0.2) * (0.8 + 重要性 * 0.4)`。
  - 时间近因性使用指数衰减，并保留最低 0.1 基础分。

## 当前妥协

- `SemanticMemory` 暂不做 LLM 自动概念抽取。
  - 优先使用 `metadata["concepts"]`。
  - 如果调用方没有传 concepts，仅用简单正则分词作为兜底。
- 图检索第一版只对 Milvus 返回的候选记忆计算图相似度。
  - 还没有单独从 Neo4j 扩展图邻居并加入候选集。
- `EpisodicMemory.add()` 和 `SemanticMemory.add()` 仍不是严格事务式双写。
  - 如果结构化/图存储写入成功但 Milvus 写入失败，暂未自动回滚。
- `MemoryTool` 默认只启用 `working`。
  - `episodic`、`semantic` 需要显式传入 `memory_types`，避免默认初始化时强制要求数据库和 embedding 配置。
- `MemoryTool` 的 `summary`、`stats`、`update`、`remove`、`forget`、`consolidate`、`clear_all` 仍是占位。
- `PerceptualMemory` 暂未接入真实多模态 embedding 模型。
  - 图像第一版使用 `caption` 或 `ocr_text` 作为文本代理。
  - 音频第一版使用 `transcript` 作为文本代理。
  - 跨模态检索当前是代理文本向量检索，不是 CLIP/CLAP 这类统一向量空间的真实跨模态检索。
- 中文分词仍比较粗糙，`WorkingMemory` 和 `SemanticMemory` 的关键词兜底都还不是正式分词器。

## 后续建议

- 增加语义概念抽取器接口，并接入 LLM/规则混合抽取。
- 为 Neo4j 实现图扩展检索，例如一跳/两跳概念邻居、关系类型权重、路径长度衰减。
- 为 Milvus 双写失败增加补偿机制或 outbox。
- 为 `MemoryTool` 补齐删除、更新、统计、整合等 action。
- 为 `PerceptualMemory` 接入真实图像/音频 embedding provider，并实现跨模态分数归一化。
- 引入更适合中文的 tokenizer，改善关键词检索和概念兜底质量。
