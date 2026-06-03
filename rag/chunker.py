"""Structure-aware Markdown chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

from .models import RagChunk


@dataclass(frozen=True)
class MarkdownBlock:
    text: str
    heading_path: list[str]
    char_start: int
    char_end: int


class MarkdownChunker:
    def __init__(
        self,
        target_tokens: int = 500,
        max_tokens: int = 800,
        overlap_tokens: int = 80,
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, markdown: str, document_id: str) -> list[RagChunk]:
        blocks = self._parse_blocks(markdown)
        chunk_texts: list[tuple[str, list[str], int, int]] = []
        current: list[MarkdownBlock] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = self._count_tokens(block.text)
            if current and current_tokens + block_tokens > self.target_tokens:
                chunk_texts.append(self._build_chunk_text(current))
                current = self._overlap_blocks(current)
                current_tokens = sum(self._count_tokens(item.text) for item in current)
            current.append(block)
            current_tokens += block_tokens

            while current_tokens > self.max_tokens and len(current) == 1:
                oversized = current.pop()
                for split in self._split_oversized_block(oversized):
                    chunk_texts.append(self._build_chunk_text([split]))
                current_tokens = 0

        if current:
            chunk_texts.append(self._build_chunk_text(current))

        chunks: list[RagChunk] = []
        for index, (content, heading_path, char_start, char_end) in enumerate(chunk_texts):
            chunks.append(
                RagChunk(
                    id=str(uuid4()),
                    document_id=document_id,
                    chunk_index=index,
                    content=content,
                    heading_path=heading_path,
                    token_count=self._count_tokens(content),
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        return chunks

    def _parse_blocks(self, markdown: str) -> list[MarkdownBlock]:
        heading_stack: list[tuple[int, str]] = []
        blocks: list[MarkdownBlock] = []
        position = 0
        parts = re.split(r"\n\s*\n", markdown)

        for part in parts:
            raw = part.strip()
            start = markdown.find(part, position)
            end = start + len(part)
            position = end
            if not raw:
                continue
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", raw)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                heading_stack = [(lvl, text) for lvl, text in heading_stack if lvl < level]
                heading_stack.append((level, title))
                continue
            heading_path = [text for _, text in heading_stack]
            blocks.append(
                MarkdownBlock(
                    text=raw,
                    heading_path=heading_path,
                    char_start=max(0, start),
                    char_end=max(0, end),
                )
            )
        return blocks

    def _build_chunk_text(self, blocks: list[MarkdownBlock]) -> tuple[str, list[str], int, int]:
        heading_path = blocks[-1].heading_path if blocks else []
        prefix = "\n".join(f"{'#' * (index + 1)} {heading}" for index, heading in enumerate(heading_path))
        body = "\n\n".join(block.text for block in blocks)
        content = f"{prefix}\n\n{body}".strip() if prefix else body
        return content, heading_path, blocks[0].char_start, blocks[-1].char_end

    def _overlap_blocks(self, blocks: list[MarkdownBlock]) -> list[MarkdownBlock]:
        selected: list[MarkdownBlock] = []
        total = 0
        for block in reversed(blocks):
            selected.append(block)
            total += self._count_tokens(block.text)
            if total >= self.overlap_tokens:
                break
        return list(reversed(selected))

    def _split_oversized_block(self, block: MarkdownBlock) -> list[MarkdownBlock]:
        tokens = self._tokenize(block.text)
        result: list[MarkdownBlock] = []
        step = max(1, self.max_tokens - self.overlap_tokens)
        for start_index in range(0, len(tokens), step):
            token_slice = tokens[start_index : start_index + self.max_tokens]
            result.append(
                MarkdownBlock(
                    text=" ".join(token_slice),
                    heading_path=block.heading_path,
                    char_start=block.char_start,
                    char_end=block.char_end,
                )
            )
        return result

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def _tokenize(self, text: str) -> list[str]:
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        non_chinese = re.sub(r"[\u4e00-\u9fff]", " ", text)
        words = re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", non_chinese)
        return chinese_chars + words
