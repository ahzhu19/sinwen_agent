"""记忆系统（语义图、概念抽取等）提示词。"""

from __future__ import annotations

from typing import Any

from .render import render_prompt

# Neo4j 语义图：SemanticMemory -[:MENTIONS]-> Concept，Concept 间可有 RELATES_TO
CONCEPT_EXTRACTION_SYSTEM_PROMPT = """你是「语义记忆」知识图谱的概念标注器。下游会把概念写入 Neo4j，用于跨条记忆的图检索与向量检索融合。

## 任务
从给定正文中抽取 {max_concepts} 个以内的**概念标签**（concept），供图谱节点与 MENTIONS 边使用。

## 概念应满足
1. **形式**：名词或名词短语，通常 2–12 个字符（英文技术名可更长）；不要整句、不要动词开头、不要代词/叹词。
2. **类型**（按需覆盖，不必全有）：
   - 实体/对象（人名、产品、项目名）
   - 技术/工具（PostgreSQL、Milvus、Neo4j）
   - 领域主题（机器学习、记忆系统）
   - 用户偏好或长期规则（深色主题、回答简洁）
   - 可复用的抽象主题（数据库迁移、架构设计）
3. **语言**：中文陈述优先用简体中文标签；专有名词、库名、协议名保留常见英文写法。
4. **去重与粒度**：同一含义只保留一个标签；不要拆得过碎（「用户」「偏好」「深色」「主题」）也不要过大（「计算机科学全部」）。
5. **禁止**：纯停用词、纯数字、无检索价值的泛词（「东西」「问题」「情况」）、与正文无关的臆造概念。

## 输出（必须严格遵守）
- 只输出一行合法 JSON，不要 markdown、不要代码块、不要解释。
- 格式：{{"concepts": ["标签1", "标签2"]}}
- 按与图谱链接、检索的价值从高到低排序；不足 {max_concepts} 个时少输出即可。"""


CONCEPT_EXTRACTION_USER_PROMPT_TEMPLATE = """请为下列「语义记忆」正文抽取概念标签。

## 正文
{content}"""


def build_concept_extraction_messages(
    content: str,
    *,
    max_concepts: int,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """组装概念抽取 Chat messages（system + user）。"""
    _ = metadata  # 预留：未来可把 session_id、source 等写入 user 段
    system_content = render_prompt(
        CONCEPT_EXTRACTION_SYSTEM_PROMPT,
        max_concepts=max_concepts,
    )
    user_content = render_prompt(
        CONCEPT_EXTRACTION_USER_PROMPT_TEMPLATE,
        content=content.strip(),
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
