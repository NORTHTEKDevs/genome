"""Journal + deterministic replay: versioned memory.

Because GENOME's write path is deterministic, a memory store can do something no
LLM-ingest system can: record every mutation and later *reproduce itself*,
provably. That unlocks:

- **Reproduce**: ``replay_journal(path)`` rebuilds the store; ``verify_journal``
  proves the live store matches its own history.
- **Roll back**: ``replay_journal(path, until_seq=N)`` is the store as it was.
- **Branch**: replay a prefix into a different storage target and diverge -
  memory for a what-if run that never contaminates the real store.

**Where the journal sits.** At the store boundary, AFTER extraction. If an LLM
extractor produced the facts, its nondeterminism happened before the journal line
was written - so replay is deterministic for every configuration, not just the
default zero-LLM path.

**What "reproduce" means, precisely.** Content, ids, scope, timestamps, metadata,
operators, parents, and graph edges are reproduced exactly; ``snapshot_hash``
covers those. Embeddings are re-derived from content on replay rather than
journaled (they are float arrays tied to a model version, and storing them would
bloat the journal ~10x). For atomic memories re-embedding is identical by
determinism of the local model; for synthesized records the replayed embedding is
the content embedding, not the original recombined vector - stated here rather
than discovered. Access statistics (accessed_at / access_count) are read-side
state, not memory state, and are excluded from the hash.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from genome.memory.graph import MemoryEdge
from genome.memory.schema import MemoryRecord
from genome.memory.store import MemoryStore


def _dumps(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _SidecarLock:
    """Cross-process mutex via an atomically-created sidecar `.lock` file.

    Portable (no platform-specific APIs) and, crucially, it locks a SEPARATE file
    so re-reading the journal's own tail under the lock never deadlocks against a
    mandatory byte-range lock on the data file. ``os.O_CREAT | O_EXCL`` is atomic
    on POSIX and Windows; the loser spins until the holder releases.
    """

    def __init__(self, target: Path) -> None:
        self._path = target.with_name(target.name + ".lock")
        self._fd: int | None = None

    def __enter__(self) -> _SidecarLock:
        import os
        import time

        deadline_spins = 0
        while True:
            try:
                self._fd = os.open(
                    str(self._path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                )
                return self
            except FileExistsError:
                deadline_spins += 1
                if deadline_spins > 5000:  # ~5s at 1ms; stale lock, break in
                    try:
                        self._path.unlink()
                    except OSError:
                        pass
                time.sleep(0.001)

    def __exit__(self, *_exc: object) -> None:
        import os

        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self._path.unlink()
        except OSError:
            pass


def _last_seq_in_file(path: Path) -> int:
    """The highest seq already in the journal, read from its tail.

    Read under the OS lock so the sequence is correct even when another process
    holds a Journal on the same path. A single memory line is bounded at ~100 KB
    (MemoryRecord.MAX_CONTENT_LEN), so a 512 KB tail always contains the last
    complete line.
    """
    if not path.is_file():
        return 0
    size = path.stat().st_size
    if size == 0:
        return 0
    with open(path, "rb") as handle:
        handle.seek(max(0, size - 512 * 1024))
        tail = handle.read()
    for line in reversed(tail.splitlines()):
        if line.strip():
            try:
                return int(json.loads(line)["seq"])
            except (ValueError, KeyError):
                continue
    return 0


class Journal:
    """Append-only JSONL of store mutations, with a monotonic sequence.

    Concurrency: the sequence is assigned under both an in-process lock and an
    OS-level exclusive file lock, re-deriving the next value from the file's own
    tail, so two Journal objects on the same path (even in different processes)
    never assign a duplicate seq. Correctness beats throughput for an audit-style
    journal, so every append pays for the lock.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, op: dict[str, Any]) -> None:
        with self._lock, _SidecarLock(self.path):
            seq = _last_seq_in_file(self.path) + 1
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(_dumps({"seq": seq, **op}) + "\n")
                handle.flush()

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        ops = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    ops.append(json.loads(line))
        return ops


class JournalingStore(MemoryStore):
    """Wraps any MemoryStore and journals every mutation. Reads pass through.

    Sitting at the store boundary means every higher-level operation - facts,
    entities, synthesis, conflict-resolution updates - is captured without any
    feature-specific journaling code, by construction.
    """

    def __init__(self, inner: MemoryStore, journal: Journal) -> None:
        self.inner = inner
        self.journal = journal

    # -- mutations (journaled) ----------------------------------------------

    def add(self, record: MemoryRecord) -> MemoryRecord:
        stored = self.inner.add(record)
        self.journal.append(
            {
                "op": "add",
                "id": stored.id,
                "content": stored.content,
                "user_id": stored.user_id,
                "agent_id": stored.agent_id,
                "created_at": stored.created_at,
                "parents": list(stored.parents),
                "operator": stored.operator,
                "metadata": stored.metadata,
            }
        )
        return stored

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        embedding: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord | None:
        result = self.inner.update(
            memory_id, content=content, embedding=embedding, metadata=metadata
        )
        if result is not None:
            self.journal.append(
                {
                    "op": "update",
                    "id": memory_id,
                    "content": content,
                    "metadata": metadata,
                    # Record whether the caller supplied a fresh embedding. On
                    # replay we must reproduce THAT choice: a content update with
                    # re_embed=False (embedding is None) keeps the old embedding
                    # live, so replay must not silently re-derive a different one.
                    "reembed": embedding is not None,
                }
            )
        return result

    def delete(self, memory_id: str) -> bool:
        deleted = self.inner.delete(memory_id)
        if deleted:
            self.journal.append({"op": "delete", "id": memory_id})
        return deleted

    def add_edge(self, edge: MemoryEdge) -> MemoryEdge:
        stored = self.inner.add_edge(edge)
        self.journal.append(
            {
                "op": "add_edge",
                "id": stored.id,
                "from_id": stored.from_id,
                "to_id": stored.to_id,
                "relation": stored.relation,
                "weight": stored.weight,
                "created_at": stored.created_at,
                "metadata": stored.metadata,
            }
        )
        return stored

    def delete_edge(self, edge_id: str) -> bool:
        deleted = self.inner.delete_edge(edge_id)
        if deleted:
            self.journal.append({"op": "delete_edge", "id": edge_id})
        return deleted

    def delete_edges_touching(self, memory_id: str) -> int:
        count = self.inner.delete_edges_touching(memory_id)
        if count:
            self.journal.append({"op": "delete_edges_touching", "id": memory_id})
        return count

    # -- reads (pass-through) -----------------------------------------------

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self.inner.get(memory_id)

    def search(self, query_embedding, **kwargs):
        return self.inner.search(query_embedding, **kwargs)

    def list_by_scope(self, **kwargs):
        return self.inner.list_by_scope(**kwargs)

    def count(self, **kwargs) -> int:
        return self.inner.count(**kwargs)

    def touch(self, memory_id: str) -> None:
        self.inner.touch(memory_id)

    def get_edge(self, edge_id: str):
        return self.inner.get_edge(edge_id)

    def edges_from(self, memory_id: str, relation: str | None = None):
        return self.inner.edges_from(memory_id, relation=relation)

    def edges_to(self, memory_id: str, relation: str | None = None):
        return self.inner.edges_to(memory_id, relation=relation)

    def close(self) -> None:
        self.inner.close()


# ---------------------------------------------------------------------------
# Snapshot, replay, verify
# ---------------------------------------------------------------------------


def snapshot_hash(memory: Any) -> str:
    """Canonical hash of the store's memory state.

    Covers content, ids, scope, timestamps, parents, operators, metadata, and
    edges. Excludes embeddings (model-version-bound floats) and access
    statistics (read-side state). Reading the store never changes its hash.
    """
    records = []
    for rec in sorted(memory.store.list_by_scope(), key=lambda r: r.id):
        records.append(
            {
                "id": rec.id,
                "content": rec.content,
                "user_id": rec.user_id,
                "agent_id": rec.agent_id,
                "created_at": rec.created_at,
                "parents": list(rec.parents),
                "operator": rec.operator,
                "metadata": rec.metadata,
            }
        )
    edges = []
    for rec in records:
        for edge in memory.store.edges_from(rec["id"]):
            edges.append(
                {
                    "id": edge.id,
                    "from_id": edge.from_id,
                    "to_id": edge.to_id,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    "created_at": edge.created_at,
                    "metadata": edge.metadata,
                }
            )
    edges.sort(key=lambda e: e["id"])
    return hashlib.sha256(
        _dumps({"records": records, "edges": edges}).encode("utf-8")
    ).hexdigest()


def replay_journal(
    path: str | Path,
    *,
    storage: str | Path = ":memory:",
    embedding_provider: Any = None,
    until_seq: int | None = None,
) -> Any:
    """Rebuild a Memory from its journal. ``until_seq`` replays a prefix
    (rollback / branch point); pass a different ``storage`` to branch."""
    from genome import Memory

    memory = Memory(storage=storage, embedding_provider=embedding_provider)
    for op in Journal.read(path):
        if until_seq is not None and op["seq"] > until_seq:
            break
        kind = op["op"]
        if kind == "add":
            vec = np.asarray(memory.embed.encode(op["content"]), dtype=np.float32)
            memory.store.add(
                MemoryRecord(
                    id=op["id"],
                    content=op["content"],
                    embedding=vec,
                    user_id=op["user_id"],
                    agent_id=op["agent_id"],
                    created_at=op["created_at"],
                    parents=list(op["parents"] or []),
                    operator=op["operator"],
                    metadata=op["metadata"] or {},
                )
            )
        elif kind == "update":
            content = op.get("content")
            # Re-embed only if the original update did (reembed defaults True for
            # journals written before this flag existed - their updates always
            # re-embedded, so True reproduces them).
            reembed = op.get("reembed", True)
            vec = (
                np.asarray(memory.embed.encode(content), dtype=np.float32)
                if content is not None and reembed
                else None
            )
            memory.store.update(
                op["id"], content=content, embedding=vec, metadata=op.get("metadata")
            )
        elif kind == "delete":
            memory.store.delete(op["id"])
        elif kind == "add_edge":
            memory.store.add_edge(
                MemoryEdge(
                    id=op["id"],
                    from_id=op["from_id"],
                    to_id=op["to_id"],
                    relation=op["relation"],
                    weight=op["weight"],
                    created_at=op["created_at"],
                    metadata=op["metadata"] or {},
                )
            )
        elif kind == "delete_edge":
            memory.store.delete_edge(op["id"])
        elif kind == "delete_edges_touching":
            memory.store.delete_edges_touching(op["id"])
        else:  # forward-compat: an unknown op is a hard error, not a skip
            raise ValueError(f"unknown journal op {kind!r} at seq {op['seq']}")
    return memory


def verify_journal(path: str | Path, memory: Any) -> bool:
    """Replay the journal into a scratch store and compare canonical hashes."""
    replayed = replay_journal(path, embedding_provider=memory.embed)
    try:
        return snapshot_hash(replayed) == snapshot_hash(memory)
    finally:
        replayed.close()
