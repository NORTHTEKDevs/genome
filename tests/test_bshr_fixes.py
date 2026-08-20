# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for defects found by the BSHR adversarial review pass.

Each test encodes an attack the review confirmed, so a future refactor that
reopens the hole fails here rather than in production.
"""

from __future__ import annotations

import numpy as np
import pytest

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


# -- #8b: AUTO-consolidation cannot launder quarantined content -------------


def test_consolidation_hybrid_inherits_worst_parent_trust():
    """The auto-consolidation path runs without the user asking, so an unguarded
    hybrid would silently re-admit poisoned content that quarantine excluded."""
    from genome.memory.consolidation import consolidate

    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    # Two low-fitness records that will be pruned together: one poisoned.
    m.add("web sourced claim alpha", user_id="u1", provenance="web")
    m.add("web sourced claim beta", user_id="u1", provenance="web")
    for i in range(4):
        m.add(f"trusted user memory {i}", user_id="u1", provenance="user")

    consolidate(
        m.store, user_id="u1", max_memories=4,
        synthesize_before_prune=True, trust_policy=m._trust_policy,
    )
    # include_quarantined: the hybrid SHOULD be quarantined, which is the point.
    hybrids = [
        r for r in m.list_all(include_quarantined=True)
        if r.metadata.get("consolidation")
    ]
    assert hybrids, "expected a consolidation hybrid to be created"
    for h in hybrids:
        assert trust_of(h, m._trust_policy) == 0, (
            "a hybrid consolidating quarantined parents must stay quarantined"
        )
    m.close()


# -- #21: graph traversal APIs respect quarantine ---------------------------


def test_related_excludes_quarantined_by_default():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    anchor = m.add("anchor memory", user_id="u1", provenance="user")[0]
    tainted = m.add("poisoned neighbour", user_id="u1", provenance="web")[0]
    m.link(anchor.id, tainted.id, "relates_to")

    neighbours = m.related(anchor.id, user_id="u1")
    assert tainted.id not in {r.id for r in neighbours}, (
        "related() must not be the side door quarantine closed in search()"
    )
    # ...but an operator can inspect what was held back, deliberately.
    seen = m.related(anchor.id, user_id="u1", include_quarantined=True)
    assert tainted.id in {r.id for r in seen}
    m.close()


def test_memories_mentioning_excludes_quarantined():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    ent = MemoryRecord(
        content="Rowan",
        embedding=np.asarray(m.embed.encode("Rowan"), dtype=np.float32),
        user_id="u1", operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": "Rowan"},
    )
    m.store.add(ent)
    tainted = m.add("Rowan is compromised", user_id="u1", provenance="web")[0]
    m.link(tainted.id, ent.id, MENTIONS)
    assert tainted.id not in {r.id for r in m.memories_mentioning(ent.id)}
    m.close()


# -- RED TEAM: list_all() leaked quarantined content ------------------------


def test_list_all_excludes_quarantined_by_default():
    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    m.add("clean user memory", user_id="u1", provenance="user")
    tainted = m.add("poisoned web memory", user_id="u1", provenance="web")[0]
    assert tainted.id not in {r.id for r in m.list_all(user_id="u1")}
    # Deliberate inspection still works.
    assert tainted.id in {
        r.id for r in m.list_all(user_id="u1", include_quarantined=True)
    }
    m.close()


# -- RED TEAM: NaN/Inf poisons the fact timeline ----------------------------


def test_record_fact_rejects_non_finite_numbers():
    m = Memory(storage=":memory:")
    ent = MemoryRecord(
        content="Sky",
        embedding=np.asarray(m.embed.encode("Sky"), dtype=np.float32),
        user_id="u1", operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": "Sky"},
    )
    m.store.add(ent)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            m.record_fact(ent.id, "score", "x", confidence=bad)
        with pytest.raises(ValueError, match="finite"):
            m.record_fact(ent.id, "score", "x", valid_from=bad)
    m.close()


# -- RED TEAM: journal tamper-evidence and oversized lines ------------------


def test_journal_detects_deleted_cancelling_pair(tmp_path):
    """Deleting a record's add AND its delete leaves final state identical, so a
    state-only check passes. The line hash chain must still catch it."""
    from genome.journal import verify_journal, verify_journal_integrity

    path = tmp_path / "j.journal"
    m = Memory(storage=":memory:", journal=path)
    doomed = m.add("evidence that will be erased", user_id="u1")[0]
    m.add("innocuous record", user_id="u1")
    m.delete(doomed.id)
    assert verify_journal(path, m) is True

    kept = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if doomed.id not in line
    ]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    intact, reason = verify_journal_integrity(path)
    assert intact is False, "removing a cancelling add/delete pair must be detected"
    assert "removed" in reason or "gap" in reason
    assert verify_journal(path, m) is False
    m.close()


def test_journal_seq_survives_an_oversized_record(tmp_path):
    """A single record can exceed any fixed tail-read window; the seq must not
    restart (which would produce duplicate seq numbers)."""
    import json as _json

    from genome.journal import Journal

    path = tmp_path / "big.journal"
    j = Journal(path)
    j.append({"op": "delete", "id": "mem_000000000001"})
    j.append({"op": "delete", "id": "mem_000000000002", "pad": "x" * 800_000})
    j.append({"op": "delete", "id": "mem_000000000003"})

    seqs = [_json.loads(line)["seq"] for line in path.read_text().splitlines()]
    assert seqs == [1, 2, 3], f"seq restarted after the oversized line: {seqs}"


# -- RED TEAM: cross-tenant fact injection ----------------------------------


def test_record_fact_refuses_another_tenants_entity():
    m = Memory(storage=":memory:")
    victim = MemoryRecord(
        content="Alice",
        embedding=np.asarray(m.embed.encode("Alice"), dtype=np.float32),
        user_id="tenant-b", operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": "Alice"},
    )
    m.store.add(victim)
    from genome.errors import ScopeError

    with pytest.raises(ScopeError):
        m.record_fact(
            victim.id, "reputation", "FRAUD - money laundering front",
            user_id="tenant-a",
        )
    m.close()


def test_record_fact_refuses_cross_tenant_source_memory():
    m = Memory(storage=":memory:")
    ent = MemoryRecord(
        content="Alice",
        embedding=np.asarray(m.embed.encode("Alice"), dtype=np.float32),
        user_id="tenant-b", operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": "Alice"},
    )
    m.store.add(ent)
    foreign = m.add("attacker's own note", user_id="tenant-a")[0]
    from genome.errors import ScopeError

    with pytest.raises(ScopeError):
        m.record_fact(
            ent.id, "reputation", "forged", source_memory_id=foreign.id,
            user_id="tenant-b",
        )
    m.close()


# -- RED TEAM: the firewall must be discoverable and agent-taggable ---------


def test_trust_policy_is_exported_from_the_package_root():
    import genome

    assert genome.TrustPolicy is TrustPolicy
    assert genome.PROVENANCE_KEY == PROVENANCE_KEY


def test_agent_archival_insert_can_mark_untrusted_origin():
    from genome.agent.memory import AgentMemory

    m = Memory(storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1))
    am = AgentMemory(m, user_id="u1", session_id="s1")
    am.archival_insert("scraped from a hostile page", provenance="web")
    hits = am.archival_search("scraped hostile")
    assert not hits.get("results"), "web-provenance agent content must be quarantined"
    m.close()


# -- RE-ATTACK: the firewall must be reachable from the async entry point ---


def test_async_memory_supports_the_firewall():
    """AsyncMemory is what the LangChain/LlamaIndex integrations use. If it cannot
    take a TrustPolicy, quarantine is unreachable for every async app."""
    import asyncio

    from genome.memory import AsyncMemory

    async def scenario():
        m = AsyncMemory(
            storage=":memory:", trust_policy=TrustPolicy(recall_min_trust=1)
        )
        await m.add("trusted user note", user_id="u1", provenance="user")
        await m.add("hostile scraped page", user_id="u1", provenance="web")
        hits = await m.search("hostile scraped", user_id="u1", limit=10)
        contents = [h.record.content for h in hits]
        await m.close()
        return contents

    contents = asyncio.run(scenario())
    assert not any("hostile" in c for c in contents), (
        "AsyncMemory must enforce quarantine like Memory does"
    )


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
