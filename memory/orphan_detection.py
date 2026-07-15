"""Orphan vector detection for memory retrieve paths.

When a vector store returns hits for memory IDs that no longer exist in the
metadata store (e.g. the metadata was deleted but the Milvus vector was not),
those hits are "orphan vectors".  This module provides a lightweight helper to
detect and log them without interrupting the retrieve flow.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def detect_orphan_vectors(
    hit_ids: Sequence[str],
    found_ids: Sequence[str],
    *,
    memory_kind: str = "episodic",
) -> list[str]:
    """Return the subset of *hit_ids* absent from *found_ids*.

    Each orphan ID is logged at WARNING level so operators can spot
    metadata/vector drift without crashing the retrieve path.
    """
    found_set = set(found_ids)
    orphans = [mid for mid in hit_ids if mid not in found_set]
    if orphans:
        logger.warning(
            "%s retrieve: %d orphan vector(s) detected (ids=%s) "
            "- metadata missing but vector exists",
            memory_kind,
            len(orphans),
            ", ".join(orphans[:10]),
        )
    return orphans
