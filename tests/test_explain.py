# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""explain_search: why each memory was or was not recalled, reproducibly.

Only a deterministic store can offer this: the report re-derives the exact
pipeline (dense scores, BM25 ranks, fusion, filters) and two runs agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from genome import Memory
from genome.explain import explain_search
from genome.firewall import TrustPolicy
from genome.memory.schema import MemoryRecord


@pytest.fixture
def mem():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    m.add("Alice loves pour-over coffee.", user_id="u1", provenance="user")
    m.add("Alice moved to Tokyo in 2024.", user_id="u1", provenance="user")
    m.add("Alice is a chemist.", user_id="u1", provenance="tool")
    m.add("SPAM: Alice loves attacker coffee.", user_id="u1", provenance="web")
    yield m
    m.close()


def test_report_matches_search_order(mem):
    report = explain_search(mem, "what coffee does Alice like?", user_id="u1", limit=3)
    hits = mem.search("what coffee does Alice like?", user_id="u1", limit=3)
    included = [c for c in report.candidates if c.included]
    assert [c.id for c in included] == [h.id for h in hits]
    assert included[0].final_rank == 1
    assert included[0].dense_score is not None


def test_quarantined_candidate_is_reported_not_hidden(mem):
    """The candidate must appear in the report (so the operator knows something was
    withheld) but its TEXT must be redacted - otherwise the diagnostic becomes the
    read path that quarantine exists to close."""
    report = explain_search(mem, "attacker coffee", user_id="u1", limit=5)
    held = [c for c in report.candidates if "quarantin" in c.reason.lower()]
    assert len(held) == 1
    assert held[0].included is False
    assert "attacker" not in held[0].content.lower()
    assert "redacted" in held[0].content.lower()
    # The diagnostic scores are still there - only the content is withheld.
    assert held[0].dense_score is not None


def test_beyond_limit_is_a_named_reason(mem):
    report = explain_search(mem, "Alice", user_id="u1", limit=1)
    beyond = [c for c in report.candidates if "limit" in c.reason.lower()]
    assert beyond, "candidates past the cutoff must say so, not vanish"
    assert all(not c.included for c in beyond)


def test_parent_filtered_candidate_is_explained(mem):
    base = mem.search("coffee", user_id="u1", limit=1)[0]
    child = MemoryRecord(
        content="Synthesized: Alice's coffee profile.",
        embedding=np.asarray(base.record.embedding, dtype=np.float32),
        user_id="u1",
        parents=[base.id],
        operator="test_synth",
    )
    mem.store.add(child)
    report = explain_search(mem, "coffee", user_id="u1", limit=5)
    parents = [c for c in report.candidates if "parent" in c.reason.lower()]
    assert [c.id for c in parents] == [base.id]


def test_hybrid_mode_reports_fusion_components(mem):
    report = explain_search(mem, "Tokyo 2024", user_id="u1", limit=3, mode="hybrid")
    tokyo = next(c for c in report.candidates if "Tokyo" in c.content)
    assert tokyo.included
    assert tokyo.fused_score is not None
    assert tokyo.bm25_rank is not None  # exact terms must register in BM25


def test_report_is_deterministic(mem):
    first = explain_search(mem, "what does Alice do?", user_id="u1", limit=3)
    second = explain_search(mem, "what does Alice do?", user_id="u1", limit=3)
    assert [(c.id, c.reason, c.final_rank) for c in first.candidates] == [
        (c.id, c.reason, c.final_rank) for c in second.candidates
    ]
