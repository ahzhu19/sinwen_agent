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

## 项目结构

```
core/           # Agent 基类、LLM 客户端、消息系统、配置
agents/         # 四种 Agent 实现 + 提示词模板
tools/          # 工具基类、注册表、链式执行、内置工具
memory/         # 记忆系统模块骨架（工作/情景/语义/感知记忆）
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
```

### 本地记忆基础设施（可选）

启动 Neo4j 图数据库 + Milvus 向量数据库：

```bash
docker compose up -d
docker compose ps
```

本地服务地址：

| 服务 | 地址 |
|------|------|
| Neo4j Browser | http://localhost:7474 |
| Neo4j Bolt | bolt://localhost:7687 |
| Milvus | http://localhost:19530 |
| MinIO Console | http://localhost:9001 |

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
```

### 运行测试

```bash
uv run pytest tests/ -v
```

## 设计理念

这个项目不依赖 LangChain、AutoGPT 等框架。每个 Agent 的推理循环都是显式手写的——消息历史怎么管、模型输出怎么解析、工具调用怎么在循环里组合——目的是理解 Agent 的内部运作机制，而不是学会调某个库的 API。
