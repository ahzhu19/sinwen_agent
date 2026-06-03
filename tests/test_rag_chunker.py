"""RAG Markdown chunker tests."""

from rag.chunker import MarkdownChunker


def test_chunker_preserves_heading_path_and_prefixes_content() -> None:
    markdown = "# RAG 系统\n\n## 数据处理流程\n\n标准 Markdown 文本进入分块。\n"
    chunker = MarkdownChunker(target_tokens=20, max_tokens=40, overlap_tokens=5)

    chunks = chunker.chunk(markdown, document_id="doc1")

    assert len(chunks) == 1
    assert chunks[0].heading_path == ["RAG 系统", "数据处理流程"]
    assert chunks[0].content.startswith("# RAG 系统\n## 数据处理流程")
    assert "标准 Markdown 文本进入分块。" in chunks[0].content
    assert chunks[0].chunk_index == 0


def test_chunker_splits_long_paragraphs_with_overlap() -> None:
    markdown = "# Long\n\n" + " ".join(f"token{i}" for i in range(45))
    chunker = MarkdownChunker(target_tokens=15, max_tokens=20, overlap_tokens=4)

    chunks = chunker.chunk(markdown, document_id="doc1")

    assert len(chunks) >= 3
    assert all(chunk.token_count <= 24 for chunk in chunks)
    assert chunks[0].heading_path == ["Long"]
    assert chunks[1].content.startswith("# Long")
    # step = max_tokens - overlap_tokens => second chunk starts around token16
    assert "token16" in chunks[1].content


def test_chunker_keeps_code_block_together_when_possible() -> None:
    markdown = "# Code\n\n```python\nprint('hello')\nprint('world')\n```\n"
    chunker = MarkdownChunker(target_tokens=10, max_tokens=50, overlap_tokens=2)

    chunks = chunker.chunk(markdown, document_id="doc1")

    assert len(chunks) == 1
    assert "```python\nprint('hello')\nprint('world')\n```" in chunks[0].content
