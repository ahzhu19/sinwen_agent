"""Outbox status reporting tests."""

from __future__ import annotations

from memory.outbox_status import collect_outbox_status, format_outbox_status


class FakePostgresOutbox:
    def status_counts(self) -> dict[str, dict[str, int]]:
        return {
            "episodic": {"pending": 2, "processing": 1, "dead": 0},
            "perceptual": {"pending": 0, "processing": 0, "dead": 1},
        }


class FakeSemanticOutbox:
    def outbox_status_counts(self) -> dict[str, int]:
        return {"pending": 3, "processing": 0, "dead": 1}


def test_collect_outbox_status_includes_postgres_and_semantic_counts() -> None:
    status = collect_outbox_status(
        pg_outbox=FakePostgresOutbox(),
        semantic_store=FakeSemanticOutbox(),
    )

    assert status["episodic"]["pending"] == 2
    assert status["episodic"]["processing"] == 1
    assert status["perceptual"]["dead"] == 1
    assert status["semantic"]["pending"] == 3
    assert status["semantic"]["dead"] == 1


def test_format_outbox_status_shows_actionable_summary() -> None:
    output = format_outbox_status(
        {
            "episodic": {"pending": 2, "processing": 1, "dead": 0},
            "semantic": {"pending": 0, "processing": 0, "dead": 1},
        }
    )

    assert "episodic: pending=2 processing=1 dead=0" in output
    assert "semantic: pending=0 processing=0 dead=1" in output
