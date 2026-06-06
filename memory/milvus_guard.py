"""Milvus collection 维度校验。"""

from __future__ import annotations

from typing import Any


class MilvusDimensionMismatchError(ValueError):
    """当前 embedding 维度与已有 collection 不一致。"""


def extract_collection_dimension(desc: dict[str, Any]) -> int | None:
    if "dimension" in desc:
        return int(desc["dimension"])
    fields = desc.get("fields") or []
    for field in fields:
        if field.get("name") == "vector":
            params = field.get("params") or {}
            if "dim" in params:
                return int(params["dim"])
    return None


def validate_collection_dimension(
    client: Any,
    collection_name: str,
    expected_dim: int,
) -> None:
    if not client.has_collection(collection_name):
        return
    desc = client.describe_collection(collection_name)
    existing_dim = extract_collection_dimension(desc)
    if existing_dim is None or existing_dim == expected_dim:
        return
    raise MilvusDimensionMismatchError(
        f"Milvus collection '{collection_name}' 维度为 {existing_dim}，"
        f"与当前 embedding 维度 {expected_dim} 不一致。"
        "请启用 USE_VERSIONED_MILVUS_COLLECTIONS 或运行 scripts/memory_reindex.py。"
    )
