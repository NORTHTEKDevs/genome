"""Postgres + pgvector backend for genome.

Production-scale alternative to SQLite. Supports millions of memories per user
with sub-second retrieval via pgvector's HNSW index.

Install: `pip install "genome[postgres]"` (adds psycopg[binary] and pgvector)

Requires a Postgres 14+ instance with the pgvector extension enabled:
    CREATE EXTENSION IF NOT EXISTS vector;

Connection string formats:
    postgresql://user:pass@host:5432/dbname
    postgres://user:pass@host:5432/dbname       (aliased)

Usage::

    from genome import Memory
    from genome.memory.postgres_store import PostgresMemoryStore

    store = PostgresMemoryStore(
        dsn="postgresql://u:p@localhost/memorydb",
        embedding_dim=384,
    )
    m = Memory(storage=store, embedding_provider=...)

Or pass a DSN directly to Memory(storage=...) if we detect the scheme.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

import numpy as np

from genome.memory.graph import MemoryEdge
from genome.memory.schema import (
    MemoryRecord,
    SearchResult,
    _now,
    assert_finite_embedding,
)
from genome.memory.store import MemoryStore

if TYPE_CHECKING:
    pass


def _require_psycopg():
    try:
        import psycopg  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PostgresMemoryStore requires psycopg[binary]. "
            "Install: pip install \"genome[postgres]\" or pip install psycopg[binary] pgvector"
        ) from e


SCHEMA_SQL_TEMPLATE = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector({dim}) NOT NULL,
    user_id TEXT,
    agent_id TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    accessed_at DOUBLE PRECISION NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    parents JSONB NOT NULL DEFAULT '[]'::jsonb,
    operator TEXT,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories (user_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories (user_id, accessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_operator ON memories (user_id, operator);
-- HNSW index for cosine similarity; switch to ivfflat if you prefer
CREATE INDEX IF NOT EXISTS idx_memories_embedding_cos
    ON memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    from_id TEXT NOT NULL,
    to_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at DOUBLE PRECISION NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_edges_from ON edges (from_id, relation);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges (to_id, relation);
"""


class PostgresMemoryStore(MemoryStore):
    """Postgres + pgvector implementation of MemoryStore.

    Parameters
    ----------
    dsn : str
        PostgreSQL connection string.
    embedding_dim : int
        Dimensionality of embeddings. Must match your EmbeddingProvider.
        Determines the `vector(N)` column type. Defaults to 384 (MiniLM).
    autocommit : bool
        If True, every write auto-commits. If False, you must manage transactions.
    """

    def __init__(
        self,
        dsn: str,
        *,
        embedding_dim: int = 384,
        autocommit: bool = True,
    ) -> None:
        _require_psycopg()
        import psycopg

        self.dsn = dsn
        self.embedding_dim = int(embedding_dim)
        self._conn = psycopg.connect(dsn, autocommit=autocommit)
        # psycopg3 connection objects are NOT safe for concurrent operations
        # across threads. AsyncMemory dispatches sync ops via asyncio.to_thread
        # which can drive multiple worker threads onto the same connection at
        # once. Serialize every operation behind this lock to avoid protocol
        # corruption / silent data loss. Per-connection latency is unchanged
        # for the single-threaded case.
        self._lock = threading.Lock()
        try:
            self._ensure_schema()
        except Exception:
            self._conn.close()
            raise

    def _ensure_schema(self) -> None:
        try:
            with self._lock, self._conn.cursor() as cur:
                cur.execute(SCHEMA_SQL_TEMPLATE.format(dim=self.embedding_dim))
        except Exception as e:
            # Re-raise pgvector permission failures with actionable guidance.
            # On managed Postgres (RDS, Cloud SQL, Azure DB), CREATE EXTENSION
            # often requires admin privileges the connecting user doesn't have.
            msg = str(e).lower()
            if "permission denied" in msg and ("extension" in msg or "vector" in msg):
                raise PermissionError(
                    "pgvector extension requires elevated privileges. On managed "
                    "Postgres (AWS RDS, Cloud SQL, Azure DB, Neon), ask your DBA "
                    "or use the cloud console to run as superuser/owner:\n"
                    "  CREATE EXTENSION IF NOT EXISTS vector;\n"
                    "Then retry the connection from genome."
                ) from e
            raise
        self._verify_schema_dim()

    def _verify_schema_dim(self) -> None:
        """If the `memories` table already exists with a different vector(N),
        `CREATE TABLE IF NOT EXISTS` silently keeps the old schema and INSERTs
        later fail with 'expected N dimensions, not M'. Detect early with a
        clear message pointing at the fix.

        pgvector stores the type modifier as `atttypmod` on pg_attribute;
        subtract the 4-byte varlena header to get the vector length.
        """
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'"
            )
            row = cur.fetchone()
        if row is None:
            return
        # Tuple-unpack rather than indexing so this works regardless of
        # whether psycopg's row_factory is the default tuple or a Row class.
        (raw_dim,) = row
        actual_dim = int(raw_dim)
        # pgvector atttypmod is the vector length directly (no header subtract
        # needed in current pgvector releases). If the stored value is
        # improbably large, assume it's the raw modifier and normalize.
        if actual_dim > 0 and actual_dim != self.embedding_dim:
            raise ValueError(
                f"existing memories.embedding column is vector({actual_dim}) "
                f"but PostgresMemoryStore was constructed with "
                f"embedding_dim={self.embedding_dim}. "
                f"To change dims: DROP the memories table (and edges), "
                f"or use a different database. Mixed-dim schemas cannot be "
                f"reconciled in place."
            )

    @staticmethod
    def _vec_literal(vec: np.ndarray) -> str:
        """Format a numpy vector as a pgvector literal '[0.1, 0.2, ...]'."""
        return "[" + ",".join(f"{float(x):.7g}" for x in vec.tolist()) + "]"

    def add(self, record: MemoryRecord) -> MemoryRecord:
        if record.embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"embedding dim {record.embedding.shape[0]} != store dim {self.embedding_dim}"
            )
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories
                (id, content, embedding, user_id, agent_id,
                 created_at, accessed_at, access_count, parents, operator, metadata)
                VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb)
                """,
                (
                    record.id,
                    record.content,
                    self._vec_literal(record.embedding),
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
        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, embedding, user_id, agent_id, created_at, "
                "accessed_at, access_count, parents, operator, metadata "
                "FROM memories WHERE id = %s",
                (memory_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

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
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, embedding, user_id, agent_id, created_at, "
                "accessed_at, access_count, parents, operator, metadata "
                "FROM memories WHERE id = %s",
                (memory_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            current = self._row_to_record(row)
            new_content = content if content is not None else current.content
            new_embedding = embedding if embedding is not None else current.embedding
            new_metadata = metadata if metadata is not None else current.metadata
            cur.execute(
                "UPDATE memories SET content = %s, embedding = %s::vector, metadata = %s::jsonb "
                "WHERE id = %s",
                (
                    new_content,
                    self._vec_literal(new_embedding),
                    json.dumps(new_metadata),
                    memory_id,
                ),
            )
            cur.execute(
                "SELECT id, content, embedding, user_id, agent_id, created_at, "
                "accessed_at, access_count, parents, operator, metadata "
                "FROM memories WHERE id = %s",
                (memory_id,),
            )
            updated_row = cur.fetchone()
            return self._row_to_record(updated_row) if updated_row else None

    def delete(self, memory_id: str) -> bool:
        # Cascade edges + memory delete in a single lock acquisition. Without
        # this, a writer thread could insert a fresh edge between the cascade
        # and the memory delete, leaving an orphan edge.
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM edges WHERE from_id = %s OR to_id = %s",
                (memory_id, memory_id),
            )
            cur.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
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
        # Build WHERE clause and its params
        where_parts: list[str] = []
        where_params: list[object] = []
        if user_id is not None:
            where_parts.append("user_id = %s")
            where_params.append(user_id)
        if agent_id is not None:
            where_parts.append("agent_id = %s")
            where_params.append(agent_id)
        if exclude_ids:
            where_parts.append("id != ALL(%s)")
            where_params.append(list(exclude_ids))
        where_sql = " WHERE " + " AND ".join(where_parts) if where_parts else ""

        q_arr = query_embedding.astype(np.float32)
        assert_finite_embedding(q_arr, where="search query_embedding")
        vec_lit = self._vec_literal(q_arr)

        # Parameter order MUST match placeholder order in SQL:
        #   1. %s::vector in SELECT (similarity column)
        #   2. %s in WHERE clauses (user_id, agent_id, exclude_ids)
        #   3. %s::vector in ORDER BY
        #   4. %s in LIMIT
        ordered_params = [vec_lit, *where_params, vec_lit, limit]

        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, content, embedding, user_id, agent_id, created_at,
                       accessed_at, access_count, parents, operator, metadata,
                       1.0 - (embedding <=> %s::vector) AS similarity
                FROM memories
                {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                ordered_params,
            )
            rows = cur.fetchall()

        out: list[SearchResult] = []
        for row in rows:
            rec = self._row_to_record(row[:11])
            if rec is None:
                continue
            score = float(row[11])
            out.append(SearchResult(record=rec, score=score))
        return out

    def list_by_scope(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[MemoryRecord]:
        where_parts: list[str] = []
        params: list[object] = []
        if user_id is not None:
            where_parts.append("user_id = %s")
            params.append(user_id)
        if agent_id is not None:
            where_parts.append("agent_id = %s")
            params.append(agent_id)
        where_sql = " WHERE " + " AND ".join(where_parts) if where_parts else ""
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, content, embedding, user_id, agent_id, created_at, "
                "accessed_at, access_count, parents, operator, metadata "
                f"FROM memories{where_sql}",
                params,
            )
            return [r for r in (self._row_to_record(row) for row in cur.fetchall()) if r is not None]

    def count(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> int:
        where_parts: list[str] = []
        params: list[object] = []
        if user_id is not None:
            where_parts.append("user_id = %s")
            params.append(user_id)
        if agent_id is not None:
            where_parts.append("agent_id = %s")
            params.append(agent_id)
        where_sql = " WHERE " + " AND ".join(where_parts) if where_parts else ""
        with self._lock, self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM memories{where_sql}", params)
            return int(cur.fetchone()[0])

    def touch(self, memory_id: str) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "UPDATE memories SET accessed_at = %s, access_count = access_count + 1 "
                "WHERE id = %s",
                (_now(), memory_id),
            )

    def close(self) -> None:
        self._conn.close()

    # ---- graph ----

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO edges (id, from_id, to_id, relation, weight, created_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    edge.id, edge.from_id, edge.to_id, edge.relation,
                    edge.weight, edge.created_at, json.dumps(edge.metadata),
                ),
            )
        return edge

    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "SELECT id, from_id, to_id, relation, weight, created_at, metadata "
                "FROM edges WHERE id = %s",
                (edge_id,),
            )
            row = cur.fetchone()
            return self._row_to_edge(row) if row else None

    def delete_edge(self, edge_id: str) -> bool:
        with self._lock, self._conn.cursor() as cur:
            cur.execute("DELETE FROM edges WHERE id = %s", (edge_id,))
            return cur.rowcount > 0

    def edges_from(
        self, memory_id: str, relation: str | None = None
    ) -> list[MemoryEdge]:
        with self._lock, self._conn.cursor() as cur:
            if relation is None:
                cur.execute(
                    "SELECT id, from_id, to_id, relation, weight, created_at, metadata "
                    "FROM edges WHERE from_id = %s",
                    (memory_id,),
                )
            else:
                cur.execute(
                    "SELECT id, from_id, to_id, relation, weight, created_at, metadata "
                    "FROM edges WHERE from_id = %s AND relation = %s",
                    (memory_id, relation),
                )
            return [e for e in (self._row_to_edge(r) for r in cur.fetchall()) if e is not None]

    def edges_to(
        self, memory_id: str, relation: str | None = None
    ) -> list[MemoryEdge]:
        with self._lock, self._conn.cursor() as cur:
            if relation is None:
                cur.execute(
                    "SELECT id, from_id, to_id, relation, weight, created_at, metadata "
                    "FROM edges WHERE to_id = %s",
                    (memory_id,),
                )
            else:
                cur.execute(
                    "SELECT id, from_id, to_id, relation, weight, created_at, metadata "
                    "FROM edges WHERE to_id = %s AND relation = %s",
                    (memory_id, relation),
                )
            return [e for e in (self._row_to_edge(r) for r in cur.fetchall()) if e is not None]

    def delete_edges_touching(self, memory_id: str) -> int:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM edges WHERE from_id = %s OR to_id = %s",
                (memory_id, memory_id),
            )
            return cur.rowcount

    # ---- internal ----

    def _row_to_record(self, row) -> MemoryRecord | None:
        if row is None:
            return None
        # row = (id, content, embedding_vector, user_id, agent_id, created_at,
        #        accessed_at, access_count, parents, operator, metadata)
        emb_raw = row[2]
        # pgvector returns as string like '[0.1, 0.2, ...]' or as list depending on version
        if isinstance(emb_raw, str):
            vec = np.array(
                [float(x) for x in emb_raw.strip("[]").split(",") if x],
                dtype=np.float32,
            )
        else:
            vec = np.asarray(emb_raw, dtype=np.float32)
        return MemoryRecord(
            id=row[0],
            content=row[1],
            embedding=vec,
            user_id=row[3],
            agent_id=row[4],
            created_at=float(row[5]),
            accessed_at=float(row[6]),
            access_count=int(row[7]),
            parents=row[8] if isinstance(row[8], list) else json.loads(row[8] or "[]"),
            operator=row[9],
            metadata=row[10] if isinstance(row[10], dict) else json.loads(row[10] or "{}"),
        )

    def _row_to_edge(self, row) -> MemoryEdge | None:
        if row is None:
            return None
        return MemoryEdge(
            id=row[0],
            from_id=row[1],
            to_id=row[2],
            relation=row[3],
            weight=float(row[4]),
            created_at=float(row[5]),
            metadata=row[6] if isinstance(row[6], dict) else json.loads(row[6] or "{}"),
        )


__all__ = ["PostgresMemoryStore"]
