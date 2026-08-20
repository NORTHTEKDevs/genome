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
import hmac
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from genome.memory.graph import MemoryEdge
from genome.memory.schema import MemoryRecord
from genome.memory.store import MemoryStore
from genome.observability import get_logger

_log = get_logger("journal")

#: prev_hash of the first journal line. Fixed so the genesis link is verifiable.
GENESIS_LINE_HASH = "0" * 64


class JournalCorruptionError(ValueError):
    """A journal line could not be parsed and it was not the final line.

    Distinct from a torn tail, which :meth:`Journal.read` recovers from: this
    means an operation in the middle of the log is unreadable, so no replay can
    faithfully reproduce the store.
    """


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
            # PermissionError, not just FileExistsError: on Windows a genuine race
            # between two creators of the same lock file surfaces as WinError 5.
            # Letting it escape kills the calling thread and SILENTLY DROPS the
            # journal record - the worst possible failure for an audit log.
            except (FileExistsError, PermissionError):
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


def _last_line(path: Path) -> dict[str, Any] | None:
    """The last complete journal record, read from the file's tail.

    The read window GROWS until a parseable line is found rather than assuming a
    fixed size: metadata values are not length-bounded, so a single record can be
    far larger than any constant we might pick, and guessing wrong silently
    restarts the sequence and produces duplicate seq numbers.
    """
    if not path.is_file():
        return None
    size = path.stat().st_size
    if size == 0:
        return None
    window = 64 * 1024
    while True:
        with open(path, "rb") as handle:
            handle.seek(max(0, size - window))
            chunk = handle.read()
        # Drop a leading partial line unless we are reading from the very start.
        lines = chunk.splitlines()
        if window < size and len(lines) > 1:
            lines = lines[1:]
        for line in reversed(lines):
            if line.strip():
                try:
                    return json.loads(line)
                except ValueError:
                    continue
        if window >= size:
            return None
        window *= 4


def _last_seq_in_file(path: Path) -> int:
    record = _last_line(path)
    if not record:
        return 0
    try:
        return int(record["seq"])
    except (KeyError, TypeError, ValueError):
        return 0


def _line_hash(op: dict[str, Any], prev_hash: str, key: bytes | None = None) -> str:
    """Hash of one journal record, chained to its predecessor.

    With ``key``, this is HMAC-SHA256 instead of a bare hash. That is the difference
    between tamper-EVIDENT and tamper-PROOF against an attacker who has write access
    to the journal file: an unkeyed chain can be recomputed by anyone (SHA-256 needs
    no secret), so a determined attacker edits a line and rechains the rest. Without
    the key they cannot produce a valid chain at all.
    """
    payload = {k: v for k, v in op.items() if k != "line_hash"}
    payload["prev_hash"] = prev_hash
    encoded = _dumps(payload).encode("utf-8")
    if key is not None:
        return hmac.new(key, encoded, hashlib.sha256).hexdigest()
    return hashlib.sha256(encoded).hexdigest()


class Journal:
    """Append-only JSONL of store mutations, with a monotonic sequence.

    Concurrency: the sequence is assigned under both an in-process lock and an
    OS-level exclusive file lock, re-deriving the next value from the file's own
    tail, so two Journal objects on the same path (even in different processes)
    never assign a duplicate seq. Correctness beats throughput for an audit-style
    journal, so every append pays for the lock.

    **Tamper-evidence vs tamper-proof.** Every line commits to its predecessor, so a
    removed, edited, or reordered line is detectable. Unkeyed, that stops accidental
    and casual tampering but NOT an attacker with write access who simply recomputes
    the whole chain - SHA-256 requires no secret. Pass ``key`` (any bytes, kept
    outside the journal's own directory - an env var, a secrets manager, an HSM) to
    make the chain an HMAC, which such an attacker cannot forge::

        Memory(journal=Journal("mem.journal", key=os.environb[b"GENOME_JOURNAL_KEY"]))

    Verification must then supply the same key. A journal written with a key and
    verified without one FAILS loudly rather than silently degrading.
    """

    def __init__(self, path: str | Path, *, key: bytes | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if key is not None and not isinstance(key, (bytes, bytearray)):
            raise TypeError(
                f"journal key must be bytes, got {type(key).__name__}. Encode a "
                "string key explicitly so the byte representation is unambiguous."
            )
        self._key = bytes(key) if key is not None else None
        self._lock = threading.Lock()

    def append(self, op: dict[str, Any]) -> None:
        with self._lock, _SidecarLock(self.path):
            previous = _last_line(self.path)
            seq = int(previous["seq"]) + 1 if previous else 1
            prev_hash = (previous or {}).get("line_hash") or GENESIS_LINE_HASH
            record = {"seq": seq, **op}
            record["line_hash"] = _line_hash(record, prev_hash, self._key)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(_dumps(record) + "\n")
                handle.flush()

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        """Parse the journal, tolerating a torn final line.

        A malformed LAST line is dropped with a warning. That is the signature of
        a crash mid-append -- a partially flushed line -- and the prefix before it
        is intact and replayable. Propagating the decode error instead (as this
        once did) made the journal unreadable in precisely the situation it exists
        for: recovering after a crash.

        A malformed INTERIOR line raises :class:`JournalCorruptionError`. Skipping
        it would silently drop an operation from the replayed state, which is
        worse than refusing to replay at all. Dropping the tail is a documented
        blind spot of the hash chain either way (see
        :func:`verify_journal_integrity`); losing a middle line is not.
        """
        ops: list[dict[str, Any]] = []
        bad: tuple[int, str] | None = None
        with open(path, "rb") as handle:
            for lineno, raw in enumerate(handle, start=1):
                if bad is not None:
                    raise JournalCorruptionError(
                        f"journal line {bad[0]} of {path} is malformed and is NOT "
                        f"the final line ({bad[1]}); the journal is corrupted in "
                        "the middle rather than merely truncated by a crash, so "
                        "replaying it would silently omit an operation"
                    )
                if not raw.strip():
                    continue
                try:
                    ops.append(json.loads(raw.decode("utf-8")))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    bad = (lineno, str(exc))
        if bad is not None:
            _log.warning(
                "dropping torn final journal line",
                extra={"path": str(path), "line": bad[0], "error": bad[1]},
            )
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
        elif kind is None and "line_hash" in op:
            continue  # a chain-only record with no state effect
        else:  # forward-compat: an unknown op is a hard error, not a skip
            raise ValueError(f"unknown journal op {kind!r} at seq {op['seq']}")
    return memory


def verify_journal_integrity(
    path: str | Path,
    *,
    key: bytes | None = None,
    expect_last_hash: str | None = None,
    expect_last_seq: int | None = None,
) -> tuple[bool, str]:
    """Check the journal's own hash chain: no line EDITED, REORDERED, or removed
    from the middle.

    **What a chain cannot do alone:** it cannot detect truncation of the TAIL. Drop
    the last N lines and the surviving prefix is still perfectly self-consistent -
    there is nothing left that references what was removed. Detecting a rollback
    requires knowing where the log should end, which must come from outside the
    file: pass ``expect_last_seq``/``expect_last_hash`` from a checkpoint you store
    elsewhere (the same place you keep the HMAC key), or compare against live state
    with :func:`verify_journal`, which catches truncation because the replayed
    state no longer matches.

    This is a SEPARATE guarantee from state reproduction. Comparing replayed state
    to live state cannot detect a tampering that cancels out - deleting a record's
    add AND its later delete leaves the final state identical, so a state-only check
    reports success while two lines of the audit trail are gone. The per-line hash
    chain closes that: every line commits to its predecessor.
    """
    prev_hash = GENESIS_LINE_HASH
    expected_seq = 1
    try:
        lines = Journal.read(path)
    except JournalCorruptionError as exc:
        # This function promises a verdict, not an exception: an unreadable line
        # IS a verification failure, and callers (CLI, API) must be able to
        # report it rather than crash.
        return False, str(exc)
    except OSError as exc:
        return False, f"journal could not be read: {exc}"
    for op in lines:
        if "line_hash" not in op:
            # Written before chaining existed: verify what we can and say so.
            return True, (
                f"journal predates hash chaining (no line_hash at seq {op.get('seq')}); "
                "integrity of line ordering cannot be proven for this file"
            )
        if op.get("seq") != expected_seq:
            return False, (
                f"journal line sequence gap: expected seq {expected_seq}, found "
                f"{op.get('seq')} - a line was removed or reordered"
            )
        recomputed = _line_hash(op, prev_hash, key)
        if not hmac.compare_digest(recomputed, str(op["line_hash"])):
            return False, (
                f"journal line {op['seq']} fails its hash: the line was edited, the "
                "line before it was removed, or the wrong key was supplied"
            )
        prev_hash = op["line_hash"]
        expected_seq += 1

    last_seq = expected_seq - 1
    # Tail-truncation check, only possible against an out-of-band checkpoint.
    if expect_last_seq is not None and last_seq != expect_last_seq:
        return False, (
            f"journal ends at seq {last_seq} but the checkpoint says {expect_last_seq}: "
            f"{expect_last_seq - last_seq} line(s) were truncated from the end"
        )
    if expect_last_hash is not None and not hmac.compare_digest(
        prev_hash, expect_last_hash
    ):
        return False, (
            "journal's final line hash does not match the checkpoint: the tail was "
            "truncated or rewritten"
        )
    return True, f"journal intact: {last_seq} line(s) chained"


def verify_journal(path: str | Path, memory: Any, *, key: bytes | None = None) -> bool:
    """Prove the live store matches its own history.

    Two independent checks, both required:
      1. the journal's own hash chain is unbroken (no line removed/edited/reordered);
      2. replaying it reproduces the live store's canonical state hash.

    Supply the same ``key`` the journal was written with. Verifying a keyed journal
    without the key fails - it does not silently fall back to an unkeyed check.
    """
    intact, _reason = verify_journal_integrity(path, key=key)
    if not intact:
        return False
    replayed = replay_journal(path, embedding_provider=memory.embed)
    try:
        return snapshot_hash(replayed) == snapshot_hash(memory)
    finally:
        replayed.close()
