"""SQLite-backed memory store.

Zero-infra, good up to ~100k memories per user on a laptop. Beyond that, swap
to Postgres+pgvector or a dedicated vector DB.

Embeddings are stored as BLOBs (numpy bytes). Cosine search loads all embeddings
for the scope into memory and computes cosine in numpy -- simple and fast at our
scale.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from genome.memory.graph import MemoryEdge
from genome.memory.schema import (
    MemoryRecord,
    SearchResult,
    _now,
    assert_finite_embedding,
)
from genome.memory.store import MemoryStore

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL,
    user_id TEXT,
    agent_id TEXT,
    created_at REAL NOT NULL,
    accessed_at REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    parents TEXT NOT NULL DEFAULT '[]',
    operator TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories (user_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories (user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories (user_id, accessed_at);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges (from_id, relation);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges (to_id, relation);
"""


class SQLiteMemoryStore(MemoryStore):
    """SQLite implementation of MemoryStore.

    Parameters
    ----------
    path : str | Path
        Path to the SQLite file. Use ":memory:" for an in-memory DB (tests).
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        # check_same_thread=False because we may be called from multiple threads
        # in an agent context (e.g. AsyncMemory via asyncio.to_thread).
        # A process-wide lock serializes ALL statements (reads included) on the
        # shared connection: unlocked reads interleaving with an in-flight write
        # transaction returned torn rows (metadata column read back as None),
        # and concurrent commits race on Linux sqlite3 ("cannot commit - no
        # transaction is active").
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
        # Track the embedding dim of existing rows so add() can fail-fast on a
        # dim mismatch (e.g. user opened a MiniLM-built file with an OpenAI
        # provider). None means "no rows yet, accept whatever dim arrives first."
        self._existing_dim: int | None = self._sniff_existing_dim()

    def _sniff_existing_dim(self) -> int | None:
        """Read one row's embedding_dim if any rows exist. Used to detect
        dim mismatches when a file-based SQLite is reopened with a different
        embedding provider."""
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding_dim FROM memories LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return int(row["embedding_dim"])

    def add(self, record: MemoryRecord) -> MemoryRecord:
        rec_dim = int(record.embedding.shape[0])
        if self._existing_dim is None:
            # First record fixes the dim for the lifetime of this store.
            self._existing_dim = rec_dim
        elif rec_dim != self._existing_dim:
            raise ValueError(
                f"embedding dim mismatch: existing rows are {self._existing_dim}-d "
                f"but new record is {rec_dim}-d. Mixing dims silently corrupts "
                f"search results. Did you open a MiniLM-built file with an "
                f"OpenAI provider (or vice versa)? Use a different storage path "
                f"or DROP the memories table to start fresh."
            )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memories
                (id, content, embedding, embedding_dim, user_id, agent_id,
                 created_at, accessed_at, access_count, parents, operator, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.content,
                    record.embedding.tobytes(),
                    int(record.embedding.shape[0]),
                    record.user_id,
                    record.agent_id,
                    record.created_at,
                    record.accessed_at,
                    record.access_count,
                    json.dumps(record.parents),
                    record.operator,
                    json.dumps(record.metadata),
                ),
            )
            self._conn.commit()
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        embedding: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord | None:
        # Validate embedding shape OUTSIDE the lock (cheap CPU work,
        # raises before we touch the DB).
        if embedding is not None:
            if not isinstance(embedding, np.ndarray):
                raise TypeError("embedding must be numpy array")
            embedding = embedding.astype(np.float32)
            assert_finite_embedding(embedding, where="update embedding")
        # Read-modify-write must happen under one lock acquisition,
        # otherwise two concurrent update(memory_id, content="A") and
        # update(memory_id, metadata={"x":1}) calls will both read the
        # same `current` snapshot and the second to commit will
        # overwrite the first's patch. Lost-update race.
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            current = _row_to_record(row)
            if current is None:
                return None
            new_content = content if content is not None else current.content
            new_embedding = embedding if embedding is not None else current.embedding
            new_metadata = metadata if metadata is not None else current.metadata
            self._conn.execute(
                """
                UPDATE memories SET content = ?, embedding = ?, embedding_dim = ?,
                       metadata = ?
                WHERE id = ?
                """,
                (
                    new_content,
                    new_embedding.tobytes(),
                    int(new_embedding.shape[0]),
                    json.dumps(new_metadata),
                    memory_id,
                ),
            )
            self._conn.commit()
            updated_row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return _row_to_record(updated_row) if updated_row else None

    def delete(self, memory_id: str) -> bool:
        # Cascade edges + delete memory atomically under one lock acquisition.
        # Without this, a thread could add an edge between the cascade-delete
        # and the memory-delete, leaving an orphan edge.
        with self._lock:
            self._conn.execute(
                "DELETE FROM edges WHERE from_id = ? OR to_id = ?",
                (memory_id, memory_id),
            )
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        rows = self._select_scope(user_id=user_id, agent_id=agent_id)
        if not rows:
            return []
        q = query_embedding.astype(np.float32)
        assert_finite_embedding(q, where="search query_embedding")
        if self._existing_dim is not None and q.shape[0] != self._existing_dim:
            raise ValueError(
                f"query embedding is {q.shape[0]}-d but stored embeddings are "
                f"{self._existing_dim}-d. The EmbeddingProvider on this Memory "
                f"does not match the provider that built this store."
            )
        q_norm = np.linalg.norm(q) or 1.0
        exclude = exclude_ids or set()

        scored: list[tuple[float, MemoryRecord]] = []
        for row in rows:
            if row["id"] in exclude:
                continue
            rec = _row_to_record(row)
            if rec is None:
                continue
            e = rec.embedding
            e_norm = np.linalg.norm(e) or 1.0
            score = float((e @ q) / (e_norm * q_norm))
            scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [SearchResult(record=r, score=s) for s, r in scored[:limit]]

    def list_by_scope(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryRecord]:
        rows = self._select_scope(user_id=user_id, agent_id=agent_id)
        return [r for r in (_row_to_record(row) for row in rows) if r is not None]

    def count(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> int:
        return len(self._select_scope(user_id=user_id, agent_id=agent_id))

    def touch(self, memory_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
                (_now(), memory_id),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- graph relations (v0.4) ----

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO edges (id, from_id, to_id, relation, weight, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge.id,
                    edge.from_id,
                    edge.to_id,
                    edge.relation,
                    edge.weight,
                    edge.created_at,
                    json.dumps(edge.metadata),
                ),
            )
            self._conn.commit()
        return edge

    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM edges WHERE id = ?", (edge_id,)
            ).fetchone()
        return _row_to_edge(row) if row else None

    def delete_edge(self, edge_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def edges_from(
        self, memory_id: str, relation: str | None = None
    ) -> list[MemoryEdge]:
        with self._lock:
            if relation is None:
                rows = self._conn.execute(
                    "SELECT * FROM edges WHERE from_id = ?", (memory_id,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM edges WHERE from_id = ? AND relation = ?",
                    (memory_id, relation),
                ).fetchall()
        return [e for e in (_row_to_edge(r) for r in rows) if e is not None]

    def edges_to(
        self, memory_id: str, relation: str | None = None
    ) -> list[MemoryEdge]:
        with self._lock:
            if relation is None:
                rows = self._conn.execute(
                    "SELECT * FROM edges WHERE to_id = ?", (memory_id,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM edges WHERE to_id = ? AND relation = ?",
                    (memory_id, relation),
                ).fetchall()
        return [e for e in (_row_to_edge(r) for r in rows) if e is not None]

    def delete_edges_touching(self, memory_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM edges WHERE from_id = ? OR to_id = ?",
                (memory_id, memory_id),
            )
            self._conn.commit()
        return cur.rowcount

    # ---- internal ----

    def _select_scope(
        self, *, user_id: str | None, agent_id: str | None
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM memories WHERE 1=1"
        params: list[object] = []
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        if agent_id is not None:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        with self._lock:
            return self._conn.execute(sql, params).fetchall()


def _row_to_record(row: sqlite3.Row | None) -> MemoryRecord | None:
    if row is None:
        return None
    dim = row["embedding_dim"]
    emb = np.frombuffer(row["embedding"], dtype=np.float32, count=dim).copy()
    return MemoryRecord(
        id=row["id"],
        content=row["content"],
        embedding=emb,
        user_id=row["user_id"],
        agent_id=row["agent_id"],
        created_at=row["created_at"],
        accessed_at=row["accessed_at"],
        access_count=row["access_count"],
        parents=json.loads(row["parents"]),
        operator=row["operator"],
        metadata=json.loads(row["metadata"]),
    )


def _row_to_edge(row: sqlite3.Row | None) -> MemoryEdge | None:
    if row is None:
        return None
    return MemoryEdge(
        id=row["id"],
        from_id=row["from_id"],
        to_id=row["to_id"],
        relation=row["relation"],
        weight=row["weight"],
        created_at=row["created_at"],
        metadata=json.loads(row["metadata"]),
    )
