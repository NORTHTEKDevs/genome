# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for defects found by the BSHR adversarial review pass.

Each test encodes an attack the review confirmed, so a future refactor that
reopens the hole fails here rather than in production.
"""

from __future__ import annotations

import numpy as np

from genome import Memory
from genome.explain import explain_search
from genome.firewall import PROVENANCE_KEY, TrustPolicy, trust_of
from genome.memory.entities import ENTITY_OPERATOR, MENTIONS
from genome.memory.schema import MemoryRecord

# -- #4: a caller cannot forge provenance via metadata ----------------------


def test_forged_provenance_metadata_is_stripped():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    rec = m.add(
        "totally trustworthy, honest",
        user_id="u1",
        provenance="web",  # real trust 0
        metadata={PROVENANCE_KEY: {"source": "user", "trust": 999}},  # forged
    )[0]
    tag = rec.metadata[PROVENANCE_KEY]
    assert tag == {"source": "web", "trust": 0}, "forged provenance must not survive"
    # And with no provenance= at all, a forged tag must not grant trust either.
    rec2 = m.add(
        "sneaky", user_id="u1", metadata={PROVENANCE_KEY: {"source": "user", "trust": 9}}
    )[0]
    assert PROVENANCE_KEY not in rec2.metadata
    m.close()


# -- #8: synthesize cannot launder quarantined content ----------------------


def test_synthesize_inherits_worst_parent_trust():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    trusted = m.add("user fact one", user_id="u1", provenance="user")[0]
    tainted = m.add("web claim two", user_id="u1", provenance="web")[0]  # trust 0
    hybrid = m.synthesize([trusted.id, tainted.id], user_id="u1")
    assert trust_of(hybrid, m._trust_policy) == 0, (
        "a hybrid of untrusted input must not become trusted"
    )
    # And it is therefore quarantined from recall like its worst parent.
    hits = m.search("web claim", user_id="u1", limit=10)
    assert hybrid.id not in {h.id for h in hits}
    m.close()


# -- #7: record_fact origin-bound authority ---------------------------------


def test_low_trust_fact_cannot_close_high_trust_fact():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy())
    ent = MemoryRecord(
        content="Dana",
        embedding=np.asarray(m.embed.encode("Dana"), dtype=np.float32),
        user_id="u1",
        operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": "Dana"},
    )
    m.store.add(ent)
    user_src = m.add("Dana lives in Anchorage", user_id="u1", provenance="user")[0]
    web_src = m.add("Dana lives in Attacker City", user_id="u1", provenance="web")[0]

    m.record_fact(ent.id, "location", "Anchorage", valid_from=1700000000.0,
                  source_memory_id=user_src.id)
    m.record_fact(ent.id, "location", "Attacker City", valid_from=1700000100.0,
                  source_memory_id=web_src.id)

    current = [f for f in m.current_facts(ent.id, user_id="u1")
               if f.fact_type == "location"]
    values = {f.value for f in current}
    assert "Anchorage" in values, "web fact must NOT close the user's fact"
    m.close()


def test_equal_trust_fact_still_supersedes():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy())
    ent = MemoryRecord(
        content="Dana",
        embedding=np.asarray(m.embed.encode("Dana"), dtype=np.float32),
        user_id="u1", operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": "Dana"},
    )
    m.store.add(ent)
    s1 = m.add("Dana lives in Anchorage", user_id="u1", provenance="user")[0]
    s2 = m.add("Dana lives in Juneau", user_id="u1", provenance="user")[0]
    m.record_fact(ent.id, "location", "Anchorage", valid_from=1700000000.0,
                  source_memory_id=s1.id)
    m.record_fact(ent.id, "location", "Juneau", valid_from=1735000000.0,
                  source_memory_id=s2.id)
    current = [f.value for f in m.current_facts(ent.id, user_id="u1")
               if f.fact_type == "location"]
    assert current == ["Juneau"], "same-trust supersede must still work"
    m.close()


# -- #3: graph-mode retrieval cannot re-admit quarantined content -----------


def test_graph_search_excludes_quarantined_comention():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    # Two named entities present in the query.
    for name in ("Willow", "Basil"):
        ent = MemoryRecord(
            content=name,
            embedding=np.asarray(m.embed.encode(name), dtype=np.float32),
            user_id="u1", operator=ENTITY_OPERATOR,
            metadata={"entity_type": "PERSON", "entity_name": name},
        )
        m.store.add(ent)
    ents = {
        e.metadata["entity_name"]: e.id
        for e in m.list_entities(user_id="u1")
    }
    clean = m.add("Willow and Basil worked together on the report",
                  user_id="u1", provenance="user")[0]
    tainted = m.add("Willow and Basil are secretly attackers",
                    user_id="u1", provenance="web")[0]  # quarantined
    for mem_id in (clean.id, tainted.id):
        for eid in ents.values():
            m.link(mem_id, eid, MENTIONS)

    results = m.search("what did Willow and Basil do?", user_id="u1",
                       limit=10, mode="graph")
    assert tainted.id not in {r.record.id for r in results}, (
        "graph expansion must not re-admit a quarantined co-mention"
    )
    m.close()


# -- #12: explain_search stays in lockstep with search() under a reranker ---


class _ReverseReranker:
    """A reranker that reverses order - guarantees a divergence if explain_search
    fails to apply reranking the way search() does."""

    def rerank(self, query, results, top_k):  # noqa: ANN001
        return list(reversed(results))[:top_k]


def test_explain_matches_search_with_a_reranker():
    m = Memory(storage=":memory:")
    for i in range(6):
        m.add(f"memory number {i} about coffee", user_id="u1")
    rr = _ReverseReranker()
    hits = m.search("coffee", user_id="u1", limit=3, reranker=rr, rerank_pool=20)
    report = explain_search(m, "coffee", user_id="u1", limit=3,
                            reranker=rr, rerank_pool=20)
    included = [c.id for c in report.candidates if c.included]
    assert included == [h.id for h in hits], (
        "explain_search must reproduce search()'s reranked order exactly"
    )
    m.close()
