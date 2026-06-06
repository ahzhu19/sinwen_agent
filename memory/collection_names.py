"""Milvus collection 版本化命名。"""

from __future__ import annotations

import re


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip()).strip("_").lower()
    return slug or "default"


def versioned_collection_name(base: str, embed_model: str, vector_size: int) -> str:
    slug = slugify_model_name(embed_model)
    return f"{base}_{slug}_{vector_size}"


def resolve_collection_name(
    base: str,
    *,
    embed_model: str,
    vector_size: int,
    use_versioned: bool,
) -> str:
    if not use_versioned:
        return base
    return versioned_collection_name(base, embed_model, vector_size)
