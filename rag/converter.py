"""Document conversion for RAG ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ConvertedDocument:
    markdown: str
    title: str | None
    mime_type: str | None


class DocumentConverter(Protocol):
    def convert(self, source: str) -> ConvertedDocument:
        ...


class PlainTextConverter:
    def convert(self, source: str) -> ConvertedDocument:
        path = Path(source)
        suffix = path.suffix.lower()
        mime_type = "text/markdown" if suffix in {".md", ".markdown"} else "text/plain"
        return ConvertedDocument(
            markdown=path.read_text(encoding="utf-8"),
            title=path.name,
            mime_type=mime_type,
        )


class MarkItDownConverter:
    def __init__(self) -> None:
        try:
            from markitdown import MarkItDown
        except Exception as exc:
            raise RuntimeError("MarkItDown 未安装，无法转换文档") from exc
        self._converter = MarkItDown()

    def convert(self, source: str) -> ConvertedDocument:
        result = self._converter.convert(source)
        markdown = getattr(result, "text_content", None) or str(result)
        return ConvertedDocument(
            markdown=markdown,
            title=Path(source).name,
            mime_type=None,
        )
