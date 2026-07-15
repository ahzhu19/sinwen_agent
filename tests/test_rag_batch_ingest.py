"""RAG batch ingestion tests (directory + URL)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.chunker import MarkdownChunker
from rag.config import RagConfig
from rag.manager import RagManager
from rag.models import BatchIngestResult
from rag.storage import InMemoryRagStore
from tests.rag_fakes import FakeConverter, FakeEmbeddingProvider, FakeLLM, FakeVectorStore


def _test_config() -> RagConfig:
    return RagConfig(enable_rag_vector_outbox=False, database_url=None)


def _make_manager(markdown: str = "# Title\n\ncontent here") -> RagManager:
    return RagManager(
        config=_test_config(),
        store=InMemoryRagStore(),
        converter=FakeConverter(markdown),
        chunker=MarkdownChunker(target_tokens=20, max_tokens=40, overlap_tokens=5),
        vector_store=FakeVectorStore(),
        embedding_provider=FakeEmbeddingProvider(),
        llm=FakeLLM(),
    )


# ---------- ingest_directory ----------


def test_ingest_directory_multiple_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nalpha")
    (tmp_path / "b.md").write_text("# B\n\nbeta")
    (tmp_path / "c.md").write_text("# C\n\ngamma")

    manager = _make_manager()
    result = manager.ingest_directory(str(tmp_path))

    assert isinstance(result, BatchIngestResult)
    assert result.success_count == 3
    assert result.error_count == 0
    titles = {doc.title for doc in result.documents}
    assert titles == {"a.md", "b.md", "c.md"}


def test_ingest_directory_custom_pattern(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.txt").write_text("plain text")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("# C")

    manager = _make_manager()
    result = manager.ingest_directory(str(tmp_path), pattern="*.md")

    assert result.success_count == 1
    assert result.documents[0].title == "a.md"


def test_ingest_directory_recursive_pattern(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B")
    (tmp_path / "sub" / "c.md").write_text("# C")

    manager = _make_manager()
    result = manager.ingest_directory(str(tmp_path), pattern="**/*.md")

    assert result.success_count == 3


def test_ingest_directory_empty_dir(tmp_path: Path) -> None:
    manager = _make_manager()
    result = manager.ingest_directory(str(tmp_path))

    assert result.success_count == 0
    assert result.error_count == 0
    assert result.documents == []


def test_ingest_directory_nonexistent_path() -> None:
    manager = _make_manager()
    with pytest.raises(NotADirectoryError):
        manager.ingest_directory("/nonexistent/path/xyz")


def test_ingest_directory_collects_errors(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("# Good")

    manager = _make_manager("# Good doc\n\ncontent")
    result = manager.ingest_directory(str(tmp_path))

    assert result.success_count == 1
    assert result.error_count == 0


def test_ingest_directory_passes_metadata(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nalpha")

    manager = _make_manager()
    result = manager.ingest_directory(
        str(tmp_path), metadata={"project": "test"}
    )

    assert result.success_count == 1
    doc = result.documents[0]
    assert doc.metadata.get("project") == "test"


# ---------- ingest_url ----------


def test_ingest_url_stores_url_as_source_uri() -> None:
    manager = _make_manager("# Downloaded\n\nweb content")
    fake_response = io.BytesIO(b"# Downloaded\n\nweb content")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = fake_response
        document = manager.ingest_url("https://example.com/article.md")

    assert document.source_uri == "https://example.com/article.md"
    assert document.source_type == "url"


def test_ingest_url_passes_metadata() -> None:
    manager = _make_manager("# Web\n\ncontent")
    fake_response = io.BytesIO(b"# Web\n\ncontent")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = fake_response
        document = manager.ingest_url(
            "https://example.com/page.html", metadata={"category": "blog"}
        )

    assert document.metadata.get("category") == "blog"


def test_ingest_url_download_failure_propagates() -> None:
    manager = _make_manager()

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = OSError("connection refused")
        with pytest.raises(OSError, match="connection refused"):
            manager.ingest_url("https://example.com/fail")


def test_ingest_url_cleans_temp_file() -> None:
    manager = _make_manager("# Web\n\ncontent")
    fake_response = io.BytesIO(b"# Web\n\ncontent")

    created_paths: list[str] = []
    original_mkstemp = __import__("tempfile").mkstemp

    def tracking_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        created_paths.append(path)
        return fd, path

    with patch("urllib.request.urlopen") as mock_urlopen, patch(
        "tempfile.mkstemp", side_effect=tracking_mkstemp
    ):
        mock_urlopen.return_value.__enter__.return_value = fake_response
        manager.ingest_url("https://example.com/doc.md")

    assert created_paths, "expected a temp file to be created"
    for path in created_paths:
        assert not Path(path).exists(), f"temp file {path} should be cleaned up"


# ---------- RagTool dispatch ----------


def test_rag_tool_ingest_url_dispatches_to_manager() -> None:
    from rag.models import RagDocument
    from tools.builtin.rag_tool import RagTool

    class StubManager:
        def __init__(self) -> None:
            self.url_calls: list[tuple] = []

        def ingest_url(self, url, metadata=None):
            self.url_calls.append((url, metadata))
            return RagDocument(
                id="urldoc123456789",
                source_uri=url,
                source_type="url",
                title="web page",
                mime_type="text/html",
                content_hash="h",
                markdown="# Web",
                status="indexed",
                metadata=metadata or {},
            )

    stub = StubManager()
    tool = RagTool(rag_manager=stub)

    result = tool.execute(
        "ingest", source="https://example.com/doc.md", source_type="url"
    )

    assert "RAG 文档已摄取" in result
    assert "urldoc12" in result
    assert stub.url_calls[0] == ("https://example.com/doc.md", {})


def test_rag_tool_ingest_directory_dispatches_to_manager() -> None:
    from rag.models import RagDocument
    from tools.builtin.rag_tool import RagTool

    class StubManager:
        def __init__(self) -> None:
            self.dir_calls: list[tuple] = []
            self._doc = RagDocument(
                id="dirdoc123456789",
                source_uri="/tmp/dir/a.md",
                source_type="file",
                title="a.md",
                mime_type="text/markdown",
                content_hash="h",
                markdown="# A",
                status="indexed",
            )

        def ingest_directory(self, path, pattern="**/*.md", metadata=None):
            self.dir_calls.append((path, pattern, metadata))
            return BatchIngestResult(documents=[self._doc], errors=[])

    stub = StubManager()
    tool = RagTool(rag_manager=stub)

    result = tool.execute(
        "ingest", source="/tmp/dir", source_type="directory", pattern="*.md"
    )

    assert "目录摄取完成" in result
    assert "成功 1 篇" in result
    assert "失败 0 篇" in result
    assert "dirdoc12" in result
    assert stub.dir_calls[0] == ("/tmp/dir", "*.md", {})


def test_rag_tool_ingest_directory_shows_errors() -> None:
    from tools.builtin.rag_tool import RagTool

    class StubManager:
        def ingest_directory(self, path, pattern="**/*.md", metadata=None):
            return BatchIngestResult(documents=[], errors=["bad.md: boom"])

    tool = RagTool(rag_manager=StubManager())
    result = tool.execute("ingest", source="/tmp/dir", source_type="directory")

    assert "成功 0 篇" in result
    assert "失败 1 篇" in result
    assert "bad.md: boom" in result
