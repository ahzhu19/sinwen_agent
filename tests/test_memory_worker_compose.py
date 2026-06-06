"""Deployment wiring tests for the memory worker service."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_compose_defines_memory_worker_service() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()

    assert "memory-worker:" in compose
    assert "scripts/memory_vector_worker.py --loop" in compose
    assert "restart: unless-stopped" in compose
    assert "milvus-standalone:" in compose


def test_application_dockerfile_exists_for_worker_image() -> None:
    dockerfile = ROOT / "Dockerfile"

    assert dockerfile.exists()
    contents = dockerfile.read_text()
    assert "uv sync" in contents
    assert "scripts/memory_vector_worker.py" in contents
