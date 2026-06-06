"""记忆相关提示词测试。"""

from __future__ import annotations

from prompts.memory import (
    CONCEPT_EXTRACTION_SYSTEM_PROMPT,
    build_concept_extraction_messages,
)


def test_concept_extraction_messages_include_content_and_max() -> None:
    messages = build_concept_extraction_messages(
        "用户偏好 PostgreSQL 与 Neo4j 构建记忆系统",
        max_concepts=5,
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "5" in messages[0]["content"]
    assert "Neo4j" in messages[0]["content"] or "MENTIONS" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "PostgreSQL" in messages[1]["content"]


def test_concept_extraction_system_prompt_escapes_json_example() -> None:
    assert "{{" in CONCEPT_EXTRACTION_SYSTEM_PROMPT
    assert "concepts" in CONCEPT_EXTRACTION_SYSTEM_PROMPT
