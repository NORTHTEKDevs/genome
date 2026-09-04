# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Journal durability, ordering, and cross-process locking.

An external review (2026-08-29) raised three points against ``genome.journal``:

1. the journal line was written AFTER the store committed (write-behind), so a
   crash between the two left a committed mutation the journal never saw;
2. ``append`` flushed but never fsync'd, so a power loss could drop lines whose
   mutations the store (which does fsync on commit) had already kept;
3. ``_SidecarLock`` broke a "stale" lock after a fixed spin count, so a holder
   whose critical section outlasted that timeout had its mutex stolen mid-section.

These tests pin the corrected behaviour. The cross-process tests spawn real
interpreters: an in-process thread cannot exercise a cross-process lock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from genome import Memory
from genome.journal import Journal, verify_journal, verify_journal_integrity
from tests.memory._fake_embed import FakeEmbeddingProvider

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def mem(tmp_path):
    path = tmp_path / "memory.journal"
    m = Memory(
        storage=":memory:",
        journal=path,
        embedding_provider=FakeEmbeddingProvider(dim=16),
    )
    yield m, path
    m.close()


def _ops(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 2. durability: every line is fsync'd, not merely flushed
# ---------------------------------------------------------------------------


def test_append_fsyncs_every_line(tmp_path, monkeypatch):
    synced: list[int] = []
    real_fsync = os.fsync

    def spy(fd):
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    j = Journal(tmp_path / "j.journal")
    j.append({"op": "delete", "id": "a"})
    j.append({"op": "delete", "id": "b"})
    assert len(synced) == 2, (
        "flush() only hands the line to the OS page cache; fsync is what makes "
        "it survive a power loss"
    )


# ---------------------------------------------------------------------------
# 1. ordering: the journal line lands BEFORE the store mutates (write-ahead)
# ---------------------------------------------------------------------------


def test_journal_line_lands_before_the_store_mutates(mem):
    """When the inner store runs a mutation, the journal already holds it.

    A crash between the two may leave the journal AHEAD of the store (replay
    reproduces the intent) but never BEHIND it (a mutation with no record).
    """
    m, path = mem
    inner = m.store.inner
    seen: list[tuple[str, bool]] = []
    real_add, real_update, real_delete = inner.add, inner.update, inner.delete

    def text() -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def add(record):
        seen.append(("add", f'"id":"{record.id}"' in text()))
        return real_add(record)

    def update(memory_id, **kw):
        seen.append(("update", '"op":"update"' in text() and memory_id in text()))
        return real_update(memory_id, **kw)

    def delete(memory_id):
        seen.append(("delete", '"op":"delete"' in text() and memory_id in text()))
        return real_delete(memory_id)

    inner.add, inner.update, inner.delete = add, update, delete

    records = m.add("The journal is written ahead of the store.", user_id="u")
    target = records[0].id
    assert m.store.update(target, content="rewritten") is not None
    assert m.delete(target) is True

    kinds = [k for k, _ in seen]
    assert "add" in kinds and "update" in kinds and "delete" in kinds
    late = [k for k, journaled in seen if not journaled]
    assert not late, f"store mutated before the journal recorded it: {late}"


def test_mutations_of_a_missing_id_are_not_journaled(mem):
    """Write-ahead must not turn every no-op into a journal line."""
    m, path = mem
    assert m.store.delete("mem_missing") is False
    assert m.store.update("mem_missing", content="x") is None
    assert m.store.delete_edge("edge_missing") is False
    assert m.store.delete_edges_touching("mem_missing") == 0
    assert _ops(path) == []


def test_failed_store_add_is_cancelled_so_replay_matches_live(mem):
    """The line is durable but the mutation failed: the journal must cancel it.

    Otherwise a routine error (a dim-mismatch ValueError, a NaN embedding)
    leaves the journal permanently one phantom record ahead of the store and
    verify_journal fails forever.
    """
    m, path = mem
    m.add("this one lands", user_id="u")
    inner = m.store.inner

    def boom(record):
        raise RuntimeError("disk on fire")

    inner.add = boom
    with pytest.raises(RuntimeError, match="disk on fire"):
        m.add("this one never lands", user_id="u")

    ops = _ops(path)
    assert [o["op"] for o in ops] == ["add", "add", "delete"], ops
    assert ops[1]["id"] == ops[2]["id"], "the cancel targets the failed add"
    assert m.count(user_id="u") == 1
    assert verify_journal(path, m), "replay must reproduce the live store"


def test_failed_store_update_is_cancelled_so_replay_matches_live(mem):
    m, path = mem
    (rec,) = m.add("original content", user_id="u")
    inner = m.store.inner

    def boom(memory_id, **kw):
        raise RuntimeError("disk on fire")

    inner.update = boom
    with pytest.raises(RuntimeError, match="disk on fire"):
        m.store.update(rec.id, content="never applied")

    ops = _ops(path)
    assert [o["op"] for o in ops] == ["add", "update", "update"], ops
    assert ops[2]["content"] == "original content"
    assert m.get(rec.id).content == "original content"
    assert verify_journal(path, m)


# ---------------------------------------------------------------------------
# 3. the lock: a slow holder is waited on, never stolen from
# ---------------------------------------------------------------------------

_APPENDER = """
import sys, time
from genome.journal import Journal
path, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
j = Journal(path)
for i in range(n):
    j.append({"op": "delete", "id": f"{tag}-{i}"})
print(time.time(), flush=True)
"""

_HOLDER = """
import sys, time
from pathlib import Path
from genome.journal import _SidecarLock
path, hold = Path(sys.argv[1]), float(sys.argv[2])
with _SidecarLock(path):
    print("HELD", flush=True)
    time.sleep(hold)
print(time.time(), flush=True)
"""


def _spawn(code: str, *args) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", code, *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_ROOT,
    )


def test_concurrent_processes_never_duplicate_a_seq(tmp_path):
    path = tmp_path / "j.journal"
    procs = [_spawn(_APPENDER, path, 25, f"p{i}") for i in range(4)]
    for p in procs:
        _out, err = p.communicate(timeout=180)
        assert p.returncode == 0, err
    ops = _ops(path)
    assert [o["seq"] for o in ops] == list(range(1, 101))
    ok, reason = verify_journal_integrity(path)
    assert ok, reason


def test_a_slow_lock_holder_is_never_stolen_from(tmp_path):
    """The create-and-spin lock broke in after a fixed spin count (~8s here), so
    a holder whose critical section ran longer lost the mutex to the waiter.
    Run with GENOME_TEST_LOCK_HOLD=12 against that lock to watch it happen.

    An OS advisory lock is released by the kernel when the holder exits or
    dies, and by nothing else, so there is no timeout to misfire.
    """
    hold = float(os.environ.get("GENOME_TEST_LOCK_HOLD", "1.5"))
    path = tmp_path / "j.journal"

    holder = _spawn(_HOLDER, path, hold)
    assert holder.stdout.readline().strip() == "HELD", holder.stderr.read()
    t_seen_held = time.time()

    appender = _spawn(_APPENDER, path, 1, "late")
    a_out, a_err = appender.communicate(timeout=hold + 120)
    h_out, h_err = holder.communicate(timeout=120)
    assert appender.returncode == 0, a_err
    assert holder.returncode == 0, h_err

    released = float(h_out.strip().splitlines()[-1])
    appended = float(a_out.strip().splitlines()[-1])
    assert released - t_seen_held >= hold * 0.9, "holder did not hold as long as asked"
    assert appended >= released - 0.01, (
        f"appender finished {released - appended:.3f}s BEFORE the holder "
        "released: the lock was stolen mid-critical-section"
    )
    assert [o["seq"] for o in _ops(path)] == [1]
