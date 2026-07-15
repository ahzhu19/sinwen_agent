"""R-06: 中文分词测试 — 验证 jieba 分词提升 Working memory 检索质量。"""

from __future__ import annotations

from memory.config import MemoryConfig
from memory.service import MemoryService
from memory.tokenizer import has_jieba, tokenize
from tools.builtin.memory_tool import MemoryTool


def _service() -> MemoryService:
    return MemoryService(
        user_id="test_user",
        config=MemoryConfig(
            database_url=None,
            enable_vector_outbox=False,
            enable_persistent_vector_outbox=False,
        ),
        memory_types=["working"],
    )


def test_tokenize_chinese_words() -> None:
    """jieba 将中文按词切分，而非逐字。"""
    tokens = tokenize("机器学习是人工智能的分支")
    assert "机器" in tokens or "机器学习" in tokens
    assert "学习" in tokens
    assert "人工" in tokens or "人工智能" in tokens


def test_tokenize_filters_punctuation() -> None:
    """标点符号不应出现在 token 列表中。"""
    tokens = tokenize("你好，世界！Python。")
    assert "，" not in tokens
    assert "。" not in tokens
    assert "！" not in tokens
    assert "你好" in tokens
    assert "python" in tokens


def test_tokenize_empty_and_whitespace() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_tokenize_mixed_chinese_english() -> None:
    """中英混合文本正确分词。"""
    tokens = tokenize("使用RAG架构做知识检索")
    assert "rag" in tokens
    assert "知识" in tokens
    assert "检索" in tokens


def test_working_memory_search_with_jieba() -> None:
    """jieba 分词后，Working memory 能按中文词检索到相关记忆。"""
    service = _service()
    tool = MemoryTool(
        user_id="test_user",
        session_id="sess-jieba-1",
        memory_service=service,
    )

    tool.execute("add", content="深度学习在图像识别领域取得了突破性进展", memory_type="working", importance=0.8)
    tool.execute("add", content="自然语言处理是人工智能的重要方向", memory_type="working", importance=0.7)

    # 搜索"图像识别"应匹配第一条，不匹配第二条
    results = service.manager.search_memory("图像识别", "working", limit=5)
    assert any("深度学习" in r.content for r in results)
    assert not any("自然语言" in r.content for r in results)

    # 搜索"自然语言"应匹配第二条
    results = service.manager.search_memory("自然语言", "working", limit=5)
    assert any("自然语言" in r.content for r in results)


def test_has_jieba_available() -> None:
    """jieba 在当前环境中应可用。"""
    assert has_jieba() is True
