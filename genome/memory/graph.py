"""Memory-to-memory graph relations.

In v0.4 we support typed edges between memories. An agent can express structural
relationships the embeddings alone cannot capture:

- SUPERSEDES: a newer fact replaces an older one (use for updates / belief revision)
- CONTRADICTS: two memories conflict (surface for resolution)
- DERIVED_FROM: a conclusion drawn from evidence (track provenance beyond recombination)
- RELATES_TO: generic topical link
- CAUSES: causal chain (A caused B)

Applications:
- `Memory.related(id, relation=SUPERSEDES)` returns what supersedes a memory
- `Memory.search(...)` can filter by having or lacking a relation
- Consolidation can follow SUPERSEDES chains to drop stale ancestors
- An agent UI can surface CONTRADICTS pairs for resolution

Edges are first-class records with id, scope, metadata, and a weight in [0, 1]
expressing confidence.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _new_edge_id() -> str:
    return f"edge_{uuid.uuid4().hex[:12]}"


def _now() -> float:
    return time.time()


# Common relation types (free-form strings; these are suggested constants)
SUPERSEDES = "supersedes"
CONTRADICTS = "contradicts"
DERIVED_FROM = "derived_from"
RELATES_TO = "relates_to"
CAUSES = "causes"


@dataclass
class MemoryEdge:
    """A typed, directed, weighted edge between two memories.

    Edges are stored independently of the memories they connect. Deleting a memory
    cascades to delete edges that touch it.
    """

    from_id: str
    to_id: str
    relation: str
    id: str = field(default_factory=_new_edge_id)
    weight: float = 1.0
    created_at: float = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.from_id or not self.to_id:
            raise ValueError("edge endpoints must be non-empty")
        if not self.relation:
            raise ValueError("relation must be non-empty")
        if not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")
        if self.metadata:
            from genome.memory.schema import assert_json_serializable
            assert_json_serializable(self.metadata, where="MemoryEdge.metadata")


__all__ = [
    "MemoryEdge",
    "SUPERSEDES",
    "CONTRADICTS",
    "DERIVED_FROM",
    "RELATES_TO",
    "CAUSES",
]
