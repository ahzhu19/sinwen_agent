# sinwen_agent

从零手写 LLM Agent，不依赖 LangChain / AutoGPT。推理循环、消息管理、工具调用全部显式实现。

实际代码量：**核心 ~6000 行，测试 ~4000 行**，当前 **276 测试通过**。

## 项目结构

```
core/           LLM 客户端、Agent 基类、消息系统、配置
agents/         四种 Agent 实现 + 共享 tool_loop + 记忆运行时
tools/          工具基类、注册表、链式执行、内置工具
context/        上下文构建：Gather → Select → Structure → Compress
memory/         四种记忆 + Service 层 + Runtime Hooks + outbox 事务
rag/            MarkItDown 文档转换 → 分块 → embedding → Milvus 检索
prompts/        Agent/记忆/RAG 的 LLM 提示词
tests/          单元测试（276 passed, 3 skipped）
scripts/        真机试用脚本
```

## Agent

四种经典范式，各自独立的推理循环：

| Agent | 循环机制 | Function Calling |
|-------|---------|:---:|
| SimpleAgent | 基础对话 + 流式输出 | ✅ |
| ReActAgent | Thought → Action → Observation | ✅ |
| ReflectionAgent | 生成初稿 → 自我批评 → 改进 | ✅ |
| PlanAndSolveAgent | 规划步骤 → 逐步求解 → 汇总 | ✅ |

共享 `agents/tool_loop.py`：工具调用的解析、执行、结果注入逻辑抽成独立模块，四种 Agent 都复用它。

四种 Agent 都可通过 `with_agent_tools(enable_memory=True, enable_rag=True)` 挂载记忆和 RAG 工具。统一注册入口在 `tools/agent_registry.py`。

## 工具系统

### 内置工具

| 工具 | 实现 | 说明 |
|------|------|------|
| Calculator | `tools/builtin/calculator.py` | AST 安全求值，防注入 |
| Search | `tools/builtin/search.py` | Tavily / SerpApi 双后端 |
| MemoryTool | `tools/builtin/memory_tool.py` | 9 个 action，统一四种记忆入口 |
| RagTool | `tools/builtin/rag_tool.py` | ingest / search / answer |

### 工具链

`tools/chain.py` — ToolChain 按顺序执行多个工具，支持模板变量 `{{previous_output}}` 和异步。所有工具实现 `arun()` / `aexecute()` 异步方法。

## 记忆系统

四种记忆覆盖从秒级到天级生命周期。详细设计见 **[docs/architecture/memory.md](docs/architecture/memory.md)**。

| 类型 | 存储 | 检索 |
|------|------|------|
| Working | 内存 InMemoryStore | TF-IDF + 关键词 + 时间衰减 + 重要性 |
| Episodic | PostgreSQL + Milvus | 向量相似度 × 0.8 + 时间近因 × 0.2 |
| Semantic | Neo4j + Milvus | 向量 × 0.7 + 图概念关系 × 0.3 |
| Perceptual | Milvus 多模态 + 内存元数据 | 向量 × 0.8 + 时间近因指数衰减 × 0.2 |

**架构分层**：

- **MemoryService**：Agent 运行时稳定内部 API 边界（`memory/service.py`）
- **Runtime Hooks**：Agent 运行前后自动检索/记录记忆（`memory/hooks.py` + `agents/memory_runtime.py`）
- **MemoryTool**：LLM 显式操作的 Tool Adapter，暴露 9 个 action（`add` / `search` / `summary` / `stats` / `update` / `remove` / `forget` / `consolidate` / `clear_all`）
- **Protocols**：ABC 协议类，存储后端可 Fake 注入测试（`memory/protocols.py`）

**事务保证**：outbox 模式。Episodic 同事务写 PostgreSQL + outbox；Semantic 同事务写 Neo4j + outbox。`memory-worker` 异步消费写入 Milvus。

## RAG 知识库

`rag/` 模块流程：**文档文件** → MarkItDown 转 Markdown → 固定大小 + 语义分块 → embedding → Milvus 向量索引。PostgreSQL 保存文档、Markdown、chunk、摄取状态。

RagTool 暴露三个 action：`ingest`（摄取本地文件）、`search`（检索片段）、`answer`（检索 + LLM 生成带来源的回答）。

真机试用：

```bash
uv run python scripts/try_rag.py ingest --source <文件路径>
```

## 上下文管理

`context/` 模块为 Agent 构建结构化上下文，流水线：**Gather → Select → Structure → Compress**。

- **Gather**：从对话历史、记忆、RAG 采集 `ContextPacket`
- **Select**：按「相关性 + 新近性」评分，在 token 预算内贪心选取
- **Structure**：渲染六分区固定骨架（Role / Task / State / Evidence / Context / Output）
- **Compress**：超预算时截断或丢弃低价值 packet

`ContextBuilder` 在 Agent 运行时与 Memory Hook 配合，自动注入检索到的记忆上下文。

## 快速开始

**环境要求**：Python >= 3.12，[uv](https://docs.astral.sh/uv/)，Docker（可选）

```bash
git clone git@github.com:ahzhu19/sinwen_agent.git
cd sinwen_agent
uv sync
```

### 配置

项目根目录创建 `.env`，参考 `.env.example`。项目使用 OpenAI 兼容接口：

```env
LLM_MODEL_ID=your-model-id
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1

# 可选
TAVILY_API_KEY=your-tavily-key      # 搜索工具
EMBED_API_KEY=your-embedding-api-key # 情景/语义记忆 embedding
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 启动记忆基础设施

```bash
docker compose up -d
```

| 服务 | 地址 |
|------|------|
| PostgreSQL | `postgresql://hello_agents:hello-agents-password@localhost:55432/hello_agents` |
| Neo4j Browser | `http://localhost:7474` |
| Neo4j Bolt | `bolt://localhost:7687` |
| Milvus | `http://localhost:19530` |
| MinIO Console | `http://localhost:9001` |

`docker compose up` 同时启动 memory-worker 自动消费 outbox。

停止：`docker compose down`；清除数据：`docker compose down -v`

### 运行测试

```bash
uv run pytest tests/ -v
```

可选真机集成测试：

```bash
RUN_EPISODIC_INTEGRATION=1 uv run pytest tests/test_episodic_integration.py -v
RUN_SEMANTIC_INTEGRATION=1 uv run pytest tests/test_semantic_integration.py -v
```

### 试用脚本

```bash
uv run python scripts/try_memory.py           # 记忆系统全流程
uv run python scripts/try_memory_agent.py     # Agent + 记忆集成
uv run python scripts/try_rag.py ingest --source <文件路径>
uv run python scripts/memory_status.py        # outbox 积压状态
```

实现状态与开放问题：

- 记忆架构：[docs/architecture/memory.md](docs/architecture/memory.md)
- 问题台账：[docs/system-issues.md](docs/system-issues.md)
- 记忆模块：[memory/implementation_status.md](memory/implementation_status.md)
- RAG 模块：[rag/implementation_status.md](rag/implementation_status.md)
