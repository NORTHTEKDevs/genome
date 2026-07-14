"""Schema for the genome memory layer."""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _new_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


def assert_finite_embedding(vec: np.ndarray, *, where: str = "embedding") -> None:
    """Refuse NaN/Inf embeddings at every boundary they could enter the system.

    Centralized so insert (MemoryRecord.__post_init__), update (store.update),
    and search (store.search) all share one rule. Adversarial NaN values
    poison cosine scores silently; better to fail loud at the boundary.
    """
    if not np.isfinite(vec).all():
        raise ValueError(
            f"{where} contains NaN or Inf values; refusing to store/search. "
            f"Check the embedding model output for the input that caused this."
        )


def assert_json_serializable(meta: dict, *, where: str = "metadata") -> None:
    """Fail at construction with a clear error when metadata contains values
    that the store would later choke on (datetime, numpy scalars, custom
    classes, etc). Without this, the failure surfaces deep inside the store
    layer's `json.dumps` call -- with a generic TypeError that doesn't tell
    the caller which field is the problem."""
    import json as _json
    try:
        _json.dumps(meta)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{where} contains non-JSON-serializable value(s): {e}. "
            f"Convert datetimes/numpy/custom objects to plain types before "
            f"passing to genome."
        ) from e


@dataclass
class MemoryRecord:
    """A single memory stored in the genome memory layer.

    Every memory carries:
    - text content and its embedding (for retrieval)
    - scoping (user_id, agent_id) for multi-tenant isolation
    - temporal metadata (created_at, accessed_at, access_count) for recency / LRU / fitness
    - provenance (parents, operator) for synthesized memories -- empty for atomic ones
    - extensible metadata bag
    """

    content: str
    embedding: np.ndarray
    id: str = field(default_factory=_new_id)
    user_id: str | None = None
    agent_id: str | None = None
    created_at: float = field(default_factory=_now)
    accessed_at: float = field(default_factory=_now)
    access_count: int = 0
    parents: list[str] = field(default_factory=list)
    operator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Defensive input limits to prevent DoS from oversized payloads.
    MAX_CONTENT_LEN = 100_000          # 100 KB per memory
    MAX_USER_ID_LEN = 256
    MAX_AGENT_ID_LEN = 256
    MAX_METADATA_KEYS = 100

    def __post_init__(self) -> None:
        if not isinstance(self.embedding, np.ndarray):
            raise TypeError("embedding must be a numpy array")
        if self.embedding.dtype != np.float32:
            self.embedding = self.embedding.astype(np.float32)
        if self.embedding.ndim != 1:
            raise ValueError("embedding must be 1-D")
        assert_finite_embedding(self.embedding, where="MemoryRecord.embedding")
        # Provenance invariant (one-directional): if parents are listed, the
        # record MUST also record which operator produced it. The reverse is
        # NOT enforced because `operator` doubles as a record-type tag for
        # entity / entity_fact / raptor_summary records that legitimately
        # have no embedding-level parents.
        if self.parents and self.operator is None:
            raise ValueError(
                "parents are set but operator is None; "
                "synthesized records must record the operator that produced them."
            )
        if any((not p) or not isinstance(p, str) for p in self.parents):
            raise ValueError("parents must be a list of non-empty strings")
        if self.metadata:
            assert_json_serializable(self.metadata, where="MemoryRecord.metadata")
        if not self.content or not self.content.strip():
            raise ValueError(
                "content must be non-empty (and not only whitespace); "
                "whitespace-only records produce empty BM25 token sets that "
                "crash hybrid search."
            )
        if len(self.content) > self.MAX_CONTENT_LEN:
            raise ValueError(
                f"content exceeds max length {self.MAX_CONTENT_LEN} "
                f"(got {len(self.content)})"
            )
        if self.user_id is not None and len(self.user_id) > self.MAX_USER_ID_LEN:
            raise ValueError(f"user_id too long (max {self.MAX_USER_ID_LEN})")
        if self.agent_id is not None and len(self.agent_id) > self.MAX_AGENT_ID_LEN:
            raise ValueError(f"agent_id too long (max {self.MAX_AGENT_ID_LEN})")
        if self.metadata and len(self.metadata) > self.MAX_METADATA_KEYS:
            raise ValueError(
                f"metadata has too many keys (max {self.MAX_METADATA_KEYS})"
            )

    @property
    def is_synthesized(self) -> bool:
        return bool(self.parents)

    @property
    def age_seconds(self) -> float:
        return max(0.0, _now() - self.created_at)


@dataclass
class SearchResult:
    """A scored memory returned from a search. Score is cosine similarity [-1, 1]."""

    record: MemoryRecord
    score: float

    @property
    def content(self) -> str:
        return self.record.content

    @property
    def id(self) -> str:
        return self.record.id
