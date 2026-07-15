"""中文分词工具：优先使用 jieba，不可用时降级到正则。"""

from __future__ import annotations

import re

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

_FALLBACK_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")
_WORD_PATTERN = re.compile(r"[\w\u4e00-\u9fff]")

# jieba 首次调用需加载词典；提前预热避免首次请求延迟
if _HAS_JIEBA:
    jieba.initialize()


def tokenize(text: str) -> list[str]:
    """对文本分词，返回小写 token 列表。

    jieba 可用时做中文词级别分词；否则降级为正则按字符/单词切分。
    """
    if not text or not text.strip():
        return []

    if _HAS_JIEBA:
        # jieba.cut 对中英混合文本均有效；过滤纯空白和标点
        return [
            word.lower()
            for word in jieba.cut(text)
            if word and word.strip() and _WORD_PATTERN.search(word)
        ]

    return _FALLBACK_PATTERN.findall(text.lower())


def has_jieba() -> bool:
    """返回 jieba 是否可用（用于测试和诊断）。"""
    return _HAS_JIEBA
