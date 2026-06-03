# hello-agents

从零手写 LLM Agent 的学习项目，实现了四种经典 Agent 范式。

## 四种 Agent

| Agent | 文件 | 原理 |
|-------|------|------|
| SimpleAgent | `agents/simple_agent.py` | 基础对话，支持 Function Calling 工具调用和流式输出 |
| ReActAgent | `agents/react_agent.py` | Thought → Action → Observation 循环，文本解析驱动 |
| ReflectionAgent | `agents/reflection_agent.py` | 生成初稿 → 自我批评 → 改进，多轮迭代 |
| PlanAndSolveAgent | `agents/plan_and_solve_agent.py` | 规划步骤列表 → 逐步求解 → 汇总 |

## 工具系统

### 内置工具

- **calculator** — 基于 AST 的安全数学表达式求值
- **search** — 混合搜索，支持 Tavily / SerpApi 双后端，可按关键词或 LLM 智能路由

### 工具链与异步

- **ToolChain**（`tools/chain.py`）— 工具链管理器，按顺序执行多个工具，支持模板变量和异步执行
- **Tool / ToolRegistry** — 新增 `arun()` / `aexecute()` 异步方法
## 记忆系统

四类记忆模块，按生命周期和存储后端划分：

| 类型 | 存储 | 检索方式 |
|------|------|----------|
| Working | 内存 (InMemoryStore) | TF-IDF + 关键词 + 时间衰减 + 重要性加权 |
| Episodic | PostgreSQL + Milvus | 向量相似度 + 时间近因 + 重要性加权 |
| Semantic | Neo4j + Milvus | 向量相似度 + 概念图关系 + 重要性加权 |
| Perceptual | Milvus (多模态) + 内存元数据 | 向量相似度 + 时间近因 + 重要性加权 |

MemoryTool 提供统一入口 (`add` / `search` 等 action)，MemoryManager 按类型路由到对应模块。
配置项见 `.env.example`，实现状态见 `memory/implementation_status.md`。


## 项目结构

```
core/           # Agent 基类、LLM 客户端、消息系统、配置
agents/         # 四种 Agent 实现 + 提示词模板
tools/          # 工具基类、注册表、链式执行、内置工具
memory/         # 记忆系统（工作/情景/语义/感知记忆，PostgreSQL + Neo4j + Milvus）
tests/          # 测试用例
scripts/        # 真机试用脚本
docs/           # 设计文档与规范
```

## 快速开始

**环境要求**：Python >= 3.12，[uv](https://docs.astral.sh/uv/)，Docker（可选，记忆基础设施）

```bash
git clone git@github.com:ahzhu19/sinwen_agent.git
cd sinwen_agent
uv sync
```

在项目根目录创建 `.env`，可参考 `.env.example` 查看完整配置项。项目使用 OpenAI 兼容接口，支持任何兼容的 LLM 服务（OpenAI / Qwen / DeepSeek 等）：

```env
LLM_MODEL_ID=your-model-id
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://api.openai.com/v1

# 可选：搜索工具密钥
TAVILY_API_KEY=your-tavily-key
SERPAPI_API_KEY=your-serpapi-key

# 可选：Embedding 配置（情景/语义记忆需要）
EMBED_API_KEY=your-embedding-api-key
EMBED_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 本地记忆基础设施（可选）

启动 PostgreSQL 关系数据库 + Neo4j 图数据库 + Milvus 向量数据库：

```bash
docker compose up -d
docker compose ps
```

本地服务地址：

| 服务 | 地址 |
|------|------|
| PostgreSQL | postgresql://hello_agents:hello-agents-password@localhost:55432/hello_agents |
| Neo4j Browser | http://localhost:7474 |
| Neo4j Bolt | bolt://localhost:7687 |
| Milvus | http://localhost:19530 |
| MinIO Console | http://localhost:9001 |

记忆系统实现状态和当前妥协项见 `memory/implementation_status.md`。

停止服务：`docker compose down`，清除数据：`docker compose down -v`

### 使用示例

```python
from agents.simple_agent import SimpleAgent
from core.llm import BaseLLM

a = SimpleAgent("助手", BaseLLM())
print(a.run("解释什么是摩尔定律"))
```

也可通过脚本试用：

```bash
python scripts/try_plan_and_solve.py --task "规划并回答：如何入门 Python"
python scripts/try_reflection.py --mode code
uv run python scripts/try_memory.py
```

### 运行测试

```bash
uv run pytest tests/ -v
```

## 设计理念

这个项目不依赖 LangChain、AutoGPT 等框架。每个 Agent 的推理循环都是显式手写的——消息历史怎么管、模型输出怎么解析、工具调用怎么在循环里组合——目的是理解 Agent 的内部运作机制，而不是学会调某个库的 API。


## 项目状态

### 已完成亮点

**Agent 范式**
- 四种经典 Agent 全部实现，推理循环显式手写，不依赖 LangChain / AutoGPT
- 每种 Agent 有对应测试覆盖

**工具系统**
- Calculator 基于 Python AST 安全求值，防护注入攻击
- Search 支持 Tavily / SerpApi 双后端，可按关键词或 LLM 智能路由
- ToolChain 实现工具链顺序执行，支持模板变量和异步
- 所有工具支持 `arun()` 异步方法

**记忆系统**
- 四类记忆模块：Working / Episodic / Semantic / Perceptual
- Working 记忆：内存存储 + TF-IDF + 关键词 + 时间衰减 + 重要性加权混合检索，带 TTL 过期和容量淘汰
- Episodic 记忆：PostgreSQL 结构化存储 + Milvus 向量检索，评分公式为 `(向量相似度 × 0.8 + 时间近因 × 0.2) × (0.8 + 重要性 × 0.4)`
- Semantic 记忆：Neo4j 图存储 + Milvus 向量检索，评分公式为 `(向量相似度 × 0.7 + 图概念关系 × 0.3) × (0.8 + 重要性 × 0.4)`
- Perceptual 记忆：多模态 Milvus 路由（text/image/audio/video/file），元数据落内存，检索公式为 `(向量相似度 × 0.8 + 时间近因 × 0.2) × (0.8 + 重要性 × 0.4)`
- MemoryTool 统一入口，`add` / `search` action 可用
- MemoryManager 支持依赖注入，可替换存储后端
- 核心代码 ~1750 行，测试 ~2110 行，测试/代码比 > 1.2
- 全部存储层有 Protocol 接口 + Fake 实现，单元测试不依赖 Docker
- 配置通过环境变量加载（`.env.example` 完整）

### 未完成工作

**Agent 层面**
- SimpleAgent / ReActAgent 尚未集成记忆系统（Agent 的 run 循环里未调用 MemoryTool）
- ReflectionAgent 的自我批评可以接入语义记忆做知识沉淀
- PlanAndSolveAgent 的计划步骤可以依赖情景记忆中的历史计划复用

**语义记忆概念抽取**
- 当前仅使用 `metadata["concepts"]` 或简单正则分词兜底
- 未接入 LLM 自动概念抽取，导致 Neo4j 图检索质量受限
- 建议引入概念抽取器接口（LLM 或 jieba 分词），替代正则兜底

**双写事务性**
- EpisodicMemory / SemanticMemory 的 `add` 先写结构化/图存储再写 Milvus
- 如果第二个写失败，第一个不回滚，可能产生孤立数据
- 建议引入 outbox 模式或 saga 补偿机制

**Neo4j 图扩展检索**
- 当前图检索只对 Milvus 返回的候选记忆计算概念匹配分数
- 尚未从 Neo4j 扩展一跳/两跳概念邻居加入候选集
- 尚未利用关系类型权重和路径长度衰减

**MemoryTool 未实现的 action**
- `summary`、`stats`、`update`、`remove`、`forget`、`consolidate`、`clear_all` 为占位实现
- search 目前仅支持 episodic 和 semantic，working 的 search 需要从 WorkingMemory.retrieve 桥接

**PerceptualMemory**
- 第一版实现已支持多模态路由（text/image/audio/video/file），图像和音频当前使用文本代理（caption/transcript），尚未接入真实多模态 embedding 模型（CLIP/CLAP）
- 跨模态检索当前是代理文本向量检索，不是统一向量空间的真实跨模态检索

**其他已知问题**
- `_add_memory` 会对调用方传入的 metadata dict 执行 `.update()` 产生副作用
- MemoryTool 默认仅启用 `working` 类型，episodic/semantic 需显式传入
- 向量维度变更（更换 embedding 模型）会导致 Milvus collection 维度不匹配
- InMemoryStore 按类型索引虽然存在，但 WorkingMemory 仍保留旧模块引用，未全局统一

**跨模块集成**
- Agent ↔ MemoryTool 尚未打通（Agent 的 tool 列表中未包含 MemoryTool）
- 各 Agent 的 prompts 中未包含记忆操作指导
- 缺少端到端集成测试（Agent + MemoryTool 联调）

完整妥协项详见 [memory/implementation_status.md](memory/implementation_status.md)。
