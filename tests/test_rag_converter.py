"""RAG converter tests."""

from pathlib import Path

from rag.converter import PlainTextConverter
from tests.rag_fakes import FakeConverter


def test_plain_text_converter_reads_markdown_file(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Guide\n\nHello", encoding="utf-8")

    result = PlainTextConverter().convert(str(path))

    assert result.markdown == "# Guide\n\nHello"
    assert result.title == "guide.md"
    assert result.mime_type == "text/markdown"


def test_fake_converter_returns_configured_markdown() -> None:
    converter = FakeConverter("# Fake\n\nBody")

    result = converter.convert("/tmp/fake.pdf")

    assert result.markdown == "# Fake\n\nBody"
    assert result.title == "fake.pdf"
    assert converter.calls == ["/tmp/fake.pdf"]
