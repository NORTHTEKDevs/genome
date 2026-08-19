# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Journal + deterministic replay: memory you can reproduce, branch, and roll back.

The journal records every mutation at the store boundary - AFTER extraction - so
replay is deterministic even when an LLM extractor produced the facts: whatever
nondeterminism happened, it happened before the journal line was written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from genome import Memory
from genome.journal import replay_journal, snapshot_hash, verify_journal


@pytest.fixture
def journaled(tmp_path):
    path = tmp_path / "memory.journal"
    m = Memory(storage=":memory:", journal=path)
    yield m, path
    m.close()


def test_replay_reproduces_the_store(journaled):
    m, path = journaled
    m.add("Alice moved to Tokyo in 2024.", user_id="u1", provenance="user")
    m.add("Alice loves pour-over coffee.", user_id="u1")
    records = m.add("Bob works at Northtek.", user_id="u1")
    m.delete(records[0].id)

    replayed = replay_journal(path, embedding_provider=m.embed)
    assert snapshot_hash(replayed) == snapshot_hash(m)
    assert replayed.count() == m.count() == 2
    replayed.close()


def test_fact_and_entity_mutations_are_captured(journaled):
    m, path = journaled
    # Facts flow through store.add/update, so the journal captures them without
    # any fact-specific journaling code. Prove it end to end.
    import numpy as np

    from genome.memory.entities import ENTITY_OPERATOR
    from genome.memory.schema import MemoryRecord

    ent = MemoryRecord(
        content="Jordan",
        embedding=np.asarray(m.embed.encode("Jordan"), dtype=np.float32),
        user_id="u1",
        operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": "Jordan"},
    )
    m.store.add(ent)
    m.record_fact(ent.id, "location", "Seattle", valid_from=1700000000.0)
    m.record_fact(ent.id, "location", "Austin", valid_from=1735000000.0)

    replayed = replay_journal(path, embedding_provider=m.embed)
    assert snapshot_hash(replayed) == snapshot_hash(m)
    timeline = replayed.entity_timeline(ent.id, user_id="u1")
    assert [f.value for f in timeline] == ["Austin", "Seattle"]
    assert timeline[1].valid_until == 1735000000.0  # invalidation replayed too
    replayed.close()


def test_reading_does_not_change_the_snapshot(journaled):
    m, _path = journaled
    m.add("stable content", user_id="u1")
    before = snapshot_hash(m)
    m.search("stable", user_id="u1", limit=5)  # touches access stats
    m.get(m.list_all()[0].id)
    assert snapshot_hash(m) == before


def test_verify_detects_journal_tampering(journaled):
    m, path = journaled
    m.add("The real fact.", user_id="u1")
    assert verify_journal(path, m) is True

    lines = path.read_text(encoding="utf-8").splitlines()
    op = json.loads(lines[0])
    op["content"] = "A forged fact."
    path.write_text(json.dumps(op) + "\n", encoding="utf-8")
    assert verify_journal(path, m) is False


def test_replay_prefix_is_rollback(journaled):
    m, path = journaled
    m.add("first", user_id="u1")
    m.add("second", user_id="u1")
    m.add("third", user_id="u1")

    earlier = replay_journal(path, embedding_provider=m.embed, until_seq=2)
    assert earlier.count() == 2
    assert sorted(r.content for r in earlier.list_all()) == ["first", "second"]
    earlier.close()


def test_edge_operations_replay(journaled):
    m, path = journaled
    a = m.add("first node", user_id="u1")[0]
    b = m.add("second node", user_id="u1")[0]
    c = m.add("third node", user_id="u1")[0]
    edge_ab = m.link(a.id, b.id, "relates_to")
    m.link(b.id, c.id, "relates_to")
    m.unlink(edge_ab.id)  # exercises delete_edge

    replayed = replay_journal(path, embedding_provider=m.embed)
    assert snapshot_hash(replayed) == snapshot_hash(m)
    assert len(replayed.edges_of(b.id, direction="both", user_id="u1")) == 1
    replayed.close()


def test_delete_cascades_edges_on_replay(journaled):
    m, path = journaled
    a = m.add("node a", user_id="u1")[0]
    b = m.add("node b", user_id="u1")[0]
    m.link(a.id, b.id, "relates_to")
    m.delete(a.id)  # cascades delete_edges_touching
    replayed = replay_journal(path, embedding_provider=m.embed)
    assert snapshot_hash(replayed) == snapshot_hash(m)
    assert replayed.edges_of(b.id, direction="both", user_id="u1") == []
    replayed.close()


def test_reembed_false_update_replays_faithfully(journaled):
    m, path = journaled
    rec = m.add("original content", user_id="u1")[0]
    original_embedding = np.array(rec.embedding, copy=True)
    # Update content but keep the embedding (re_embed=False): live store keeps the
    # old vector; replay must reproduce THAT, not silently re-embed the new content.
    m.update(rec.id, content="rewritten content entirely different", re_embed=False)

    replayed = replay_journal(path, embedding_provider=m.embed)
    live = m.get(rec.id)
    rep = replayed.get(rec.id)
    assert rep.content == live.content == "rewritten content entirely different"
    assert np.allclose(rep.embedding, original_embedding), (
        "replay must keep the original embedding when re_embed was False"
    )
    replayed.close()


def test_seq_reflects_file_not_just_memory(tmp_path):
    """A second Journal on the same path continues the sequence, never restarts."""
    from genome.journal import Journal

    path = tmp_path / "j.journal"
    m1 = Memory(storage=":memory:", journal=path)
    m1.add("one", user_id="u1")
    m1.add("two", user_id="u1")
    m1.close()

    # A fresh Journal object (as a second process would create) must not reuse seq 1.
    j2 = Journal(path)
    j2.append({"op": "delete", "id": "mem_xxxxxxxxxxxx"})
    seqs = [json.loads(line)["seq"] for line in path.read_text().splitlines()]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "seqs stay unique/ordered"
    assert seqs[-1] == len(seqs)


def test_no_journal_no_file(tmp_path):
    m = Memory(storage=":memory:")
    m.add("nothing recorded", user_id="u1")
    assert list(tmp_path.iterdir()) == []
    m.close()


def test_journal_lines_are_sequenced_and_compact(journaled):
    m, path = journaled
    m.add("one", user_id="u1")
    m.add("two", user_id="u1")
    ops = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [op["seq"] for op in ops] == list(range(1, len(ops) + 1))
    assert all(op["op"] == "add" for op in ops)
    assert all("embedding" not in op for op in ops), (
        "embeddings are re-derived on replay, never journaled"
    )
