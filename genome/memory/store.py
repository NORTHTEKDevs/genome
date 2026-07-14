"""Abstract storage interface for the genome memory layer.

Subclasses implement the physical storage (SQLite, Postgres, Qdrant, ...).
The Memory facade sits on top of this and handles embedding / extraction / synthesis.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from genome.memory.graph import MemoryEdge
from genome.memory.schema import MemoryRecord, SearchResult


class MemoryStore(ABC):
    """Abstract store for MemoryRecords.

    Scoping: most methods take optional `user_id` and `agent_id` filters. When
    omitted, the method operates across all scopes. When provided, only records
    matching that exact scope are considered.
    """

    @abstractmethod
    def add(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a new record. Returns the stored record (id may be assigned)."""

    @abstractmethod
    def get(self, memory_id: str) -> MemoryRecord | None:
        """Return the record with the given id, or None."""

    @abstractmethod
    def update(self, memory_id: str, *, content: str | None = None,
               embedding: np.ndarray | None = None,
               metadata: dict | None = None) -> MemoryRecord | None:
        """Update a record in place. Any field left as None is untouched."""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a record. Returns True if something was deleted."""

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        """Cosine-similarity search within the given scope.

        `exclude_ids` lets the caller exclude specific records from results. This
        is how parent filtering is implemented -- the Memory facade passes the
        synthesized memory's parent ids here.
        """

    @abstractmethod
    def list_by_scope(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryRecord]:
        """Return all records in the given scope."""

    @abstractmethod
    def count(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> int:
        """Count records in the given scope."""

    @abstractmethod
    def touch(self, memory_id: str) -> None:
        """Mark a record as just-accessed (updates accessed_at + increments access_count)."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources (DB connections, files)."""

    # ---- graph relations (v0.4) ----

    @abstractmethod
    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        """Persist a new edge between two memories."""

    @abstractmethod
    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        """Fetch an edge by id."""

    @abstractmethod
    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge. Returns True if something was deleted."""

    @abstractmethod
    def edges_from(
        self, memory_id: str, relation: str | None = None
    ) -> list[MemoryEdge]:
        """Return outgoing edges from `memory_id`, optionally filtered by relation."""

    @abstractmethod
    def edges_to(
        self, memory_id: str, relation: str | None = None
    ) -> list[MemoryEdge]:
        """Return incoming edges to `memory_id`, optionally filtered by relation."""

    @abstractmethod
    def delete_edges_touching(self, memory_id: str) -> int:
        """Delete all edges where `memory_id` is either endpoint. Returns count deleted.

        Called automatically on memory delete for referential integrity.
        """
