# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the round-7 adversarial findings.

Theme: corrupt persisted data must degrade, not detonate. Every one of these was
a raw driver exception escaping an API that had a perfectly good verdict type to
return instead, or one bad row denying access to every healthy row beside it.
"""

import sqlite3

import numpy as np
import pytest

import genome
from genome.journal import (
    Journal,
    JournalCorruptionError,
    replay_journal,
    verify_journal_integrity,
)
from genome.memory.consolidation import consolidate
from genome.memory.sqlite_store import SQLiteMemoryStore


def _write_journal(path, n=5):
    j = Journal(path)
    for i in range(n):
        j.append({"op": "add", "record": {"id": f"m{i}", "content": f"c{i}"}})
    return j


class TestJournalTornTail:
    """A crash mid-append leaves a partial last line; replay must survive it."""

    def test_truncated_last_line_is_dropped_not_raised(self, tmp_path):
        p = tmp_path / "j.jsonl"
        _write_journal(p)
        raw = p.read_bytes()
        p.write_bytes(raw[: len(raw) - 30])  # tear the final line
        ops = Journal.read(p)
        assert len(ops) == 4
        assert all("seq" in o for o in ops)

    def test_trailing_garbage_is_dropped(self, tmp_path):
        p = tmp_path / "j.jsonl"
        _write_journal(p)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write("this is not json")
        assert len(Journal.read(p)) == 5

    def test_trailing_invalid_utf8_is_dropped(self, tmp_path):
        p = tmp_path / "j.jsonl"
        _write_journal(p)
        with open(p, "ab") as fh:
            fh.write(b"\xff\xfe not utf8")
        assert len(Journal.read(p)) == 5

    def test_interior_corruption_raises_rather_than_silently_skipping(self, tmp_path):
        """Skipping a middle line would drop an operation from the replay."""
        p = tmp_path / "j.jsonl"
        _write_journal(p)
        lines = p.read_text(encoding="utf-8").splitlines()
        lines[2] = "{corrupt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(JournalCorruptionError) as exc:
            Journal.read(p)
        assert "NOT the final line" in str(exc.value)

    def test_verify_returns_a_verdict_on_interior_corruption(self, tmp_path):
        """The (ok, reason) contract must hold; no exception may escape."""
        p = tmp_path / "j.jsonl"
        _write_journal(p)
        lines = p.read_text(encoding="utf-8").splitlines()
        lines[2] = "{corrupt"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok, reason = verify_journal_integrity(p)
        assert ok is False
        assert "malformed" in reason

    def test_replay_survives_a_torn_tail(self, tmp_path):
        """The whole point of a journal: recover the prefix after a crash."""
        p = tmp_path / "j.jsonl"
        mem = genome.Memory(storage=":memory:", journal=str(p))
        for i in range(4):
            mem.add(f"memory number {i}", user_id="u", agent_id="a")
        raw = p.read_bytes()
        p.write_bytes(raw[: len(raw) - 20])

        rebuilt = replay_journal(p)
        assert rebuilt.count(user_id="u", agent_id="a") >= 3


class TestOneBadRowDoesNotDenyTheScope:
    """A single undecodable row used to raise for the entire scope."""

    def _store_with_one_corrupt_row(self, tmp_path, corrupt):
        path = tmp_path / "m.db"
        store = SQLiteMemoryStore(str(path))
        rng = np.random.default_rng(0)
        for i in range(5):
            store.add(
                genome.memory.schema.MemoryRecord(
                    id=f"m{i}",
                    content=f"healthy record {i}",
                    embedding=rng.random(8, dtype=np.float32),
                    user_id="alice",
                    agent_id="agent",
                )
            )
        store.close()
        conn = sqlite3.connect(path)
        corrupt(conn)
        conn.commit()
        conn.close()
        return SQLiteMemoryStore(str(path))

    def test_wrong_embedding_dim_skips_only_that_row(self, tmp_path):
        store = self._store_with_one_corrupt_row(
            tmp_path,
            lambda c: c.execute("UPDATE memories SET embedding_dim=999 WHERE id='m2'"),
        )
        rows = store.list_by_scope(user_id="alice", agent_id="agent")
        assert len(rows) == 4
        assert "m2" not in {r.id for r in rows}
        hits = store.search(
            np.random.default_rng(1).random(8, dtype=np.float32),
            user_id="alice",
            agent_id="agent",
            limit=10,
        )
        assert len(hits) == 4
        store.close()

    def test_non_finite_embedding_skips_only_that_row(self, tmp_path):
        bad = np.full(8, np.nan, dtype=np.float32).tobytes()
        store = self._store_with_one_corrupt_row(
            tmp_path,
            lambda c: c.execute(
                "UPDATE memories SET embedding=? WHERE id='m2'", (bad,)
            ),
        )
        rows = store.list_by_scope(user_id="alice", agent_id="agent")
        assert len(rows) == 4
        assert "m2" not in {r.id for r in rows}
        store.close()

    def test_malformed_metadata_json_skips_only_that_row(self, tmp_path):
        store = self._store_with_one_corrupt_row(
            tmp_path,
            lambda c: c.execute("UPDATE memories SET metadata='{oops' WHERE id='m2'"),
        )
        assert len(store.list_by_scope(user_id="alice", agent_id="agent")) == 4
        store.close()

    def test_healthy_store_still_returns_everything(self, tmp_path):
        """Negative control: isolation must not drop valid rows."""
        store = self._store_with_one_corrupt_row(tmp_path, lambda c: None)
        assert len(store.list_by_scope(user_id="alice", agent_id="agent")) == 5
        store.close()


class TestConsolidateBoundValidation:
    def test_negative_max_memories_is_refused(self):
        """-1 silently became a negative slice that pruned exactly one record."""
        mem = genome.Memory(storage=":memory:")
        for i in range(4):
            mem.add(f"record {i}", user_id="u1", agent_id="a1")
        with pytest.raises(ValueError, match="max_memories must be >= 0"):
            consolidate(mem.store, user_id="u1", agent_id="a1", max_memories=-1)

    def test_zero_still_prunes_everything(self):
        """Negative control: 0 is a legitimate bound, not an error."""
        mem = genome.Memory(storage=":memory:")
        for i in range(4):
            mem.add(f"record {i}", user_id="u1", agent_id="a1")
        result = consolidate(
            mem.store, user_id="u1", agent_id="a1", max_memories=0
        )
        assert result.pruned == 4
        assert mem.count(user_id="u1", agent_id="a1") == 0


class TestEmptyJournalIsNotIntact:
    """Round 10: verifying zero lines with no checkpoint used to report intact.

    A journal that was never written and one truncated to nothing are the same
    file, so "intact" would certify the erased case.
    """

    def test_emptied_journal_is_not_reported_intact(self, tmp_path):
        p = tmp_path / "j.jsonl"
        _write_journal(p, n=3)
        p.write_bytes(b"")
        ok, reason = verify_journal_integrity(p)
        assert ok is False
        assert "empty" in reason

    def test_a_checkpoint_turns_it_into_a_proven_break(self, tmp_path):
        """With out-of-band expectations this is detected as truncation, not doubt."""
        p = tmp_path / "j.jsonl"
        _write_journal(p, n=3)
        p.write_bytes(b"")
        ok, reason = verify_journal_integrity(p, expect_last_seq=3)
        assert ok is False
        assert "truncated" in reason

    def test_a_real_journal_still_verifies(self, tmp_path):
        """Negative control."""
        p = tmp_path / "j.jsonl"
        _write_journal(p, n=3)
        ok, reason = verify_journal_integrity(p)
        assert ok is True, reason
