"""Token 估算工具。

采用与 rag/chunker 相同的分词规则：中文按单字、英文按词，
用于预算装箱时的近似 token 计数（非精确 tokenizer）。
"""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """将文本拆分为 token 列表。

    - 中文字符：每个汉字单独为一个 token
    - 非中文：按英文单词、数字、标点切分
    """
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    non_chinese = re.sub(r"[\u4e00-\u9fff]", " ", text)
    words = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", non_chinese)
    return chinese_chars + words


def estimate_tokens(text: str) -> int:
    """估算文本 token 数（len(tokenize(text))）。"""
    return len(tokenize(text))
