"""Multi-hop graph retrieval (mode='graph'), precision-first design.

Two properties that matter:
  1. REGRESSION GUARD: on a query that is NOT multi-hop (names < 2 known
     entities), graph mode returns exactly the hybrid result -- no displacement.
     (The v1 design ignored this and crushed retrieval.)
  2. When the query names >= 2 entities and a relevant co-mentioned sibling
     exists near the retrieval boundary, graph surfaces it.
"""
from __future__ import annotations

import numpy as np

from genome.memory.entities import ENTITY_OPERATOR, MENTIONS
from genome.memory.facade import Memory
from genome.memory.schema import MemoryRecord


class ControlledEmbedder:
    model_name = "controlled"

    def __init__(self, table, dim=4):
        self.table = {k: np.asarray(v, dtype=np.float32) for k, v in table.items()}
        self.dim = dim
        self._far = np.asarray([0, 0, 0, -1], dtype=np.float32)

    def encode(self, text):
        return self.table.get(text, self._far)

    def encode_batch(self, texts):
        return np.stack([self.encode(t) for t in texts])


def _mk(mem, text, vec, **kw):
    r = MemoryRecord(content=text, embedding=np.asarray(vec, dtype=np.float32),
                     user_id="u", **kw)
    mem.store.add(r)
    return r


def test_graph_regression_guard_non_multihop_equals_hybrid():
    """A query naming < 2 known entities must yield IDENTICAL results to hybrid
    -- graph mode cannot displace evidence on ordinary questions."""
    emb = {
        "where does Alice live": [1.0, 0.0, 0.0, 0.0],
        "a1": [0.95, 0.30, 0.0, 0.0],
        "a2": [0.90, 0.30, 0.0, 0.0],
        "junk co-mention": [0.02, 0.99, 0.0, 0.0],
    }
    mem = Memory(embedding_provider=ControlledEmbedder(emb))
    try:
        s = _mk(mem, "a1", emb["a1"])
        _mk(mem, "a2", emb["a2"])
        junk = _mk(mem, "junk co-mention", emb["junk co-mention"])
        # Alice entity mentioned by BOTH s and junk -- the v1 trap.
        ent = _mk(mem, "Alice", mem.embed.encode("Alice"),
                  operator=ENTITY_OPERATOR,
                  metadata={"entity_type": "PERSON", "entity_name": "Alice"})
        mem.link(s.id, ent.id, relation=MENTIONS)
        mem.link(junk.id, ent.id, relation=MENTIONS)

        dense = mem.search("where does Alice live", user_id="u", limit=2,
                           filter_parents=False, mode="dense")
        graph = mem.search("where does Alice live", user_id="u", limit=2,
                           filter_parents=False, mode="graph")
        # Only ONE named entity ("Alice") -> not multi-hop -> identical to the
        # dense base, and the low-relevance junk co-mention must NOT be pulled in.
        assert [r.record.id for r in graph] == [r.record.id for r in dense]
        assert junk.id not in {r.record.id for r in graph}
    finally:
        mem.close()


def test_graph_surfaces_relevant_sibling_on_multihop_query():
    """Query names two entities (Nate, Joanna); a boundary-relevant memory that
    mentions Joanna is surfaced even though dense/hybrid ranked it out."""
    emb = {
        "what do Nate and Joanna like": [1.0, 0.0, 0.0, 0.0],
        "Nate likes turtles":  [0.96, 0.28, 0.0, 0.0],   # base
        "d1":                  [0.92, 0.30, 0.0, 0.0],   # base
        "d2":                  [0.90, 0.30, 0.0, 0.0],   # base boundary
        "also turtles":        [0.93, 0.37, 0.0, 0.0],   # T: near-boundary, no query words
    }
    mem = Memory(embedding_provider=ControlledEmbedder(emb))
    try:
        S = _mk(mem, "Nate likes turtles", emb["Nate likes turtles"])
        _mk(mem, "d1", emb["d1"])
        _mk(mem, "d2", emb["d2"])
        T = _mk(mem, "also turtles", emb["also turtles"])
        nate = _mk(mem, "Nate", mem.embed.encode("Nate"), operator=ENTITY_OPERATOR,
                   metadata={"entity_type": "PERSON", "entity_name": "Nate"})
        joanna = _mk(mem, "Joanna", mem.embed.encode("Joanna"),
                     operator=ENTITY_OPERATOR,
                     metadata={"entity_type": "PERSON", "entity_name": "Joanna"})
        mem.link(S.id, nate.id, relation=MENTIONS)
        mem.link(T.id, joanna.id, relation=MENTIONS)  # T mentions the 2nd named entity

        dense = mem.search("what do Nate and Joanna like", user_id="u", limit=3,
                           filter_parents=False, mode="dense")
        assert T.id not in {r.record.id for r in dense}, "setup: T outside dense top-3"

        graph = mem.search("what do Nate and Joanna like", user_id="u", limit=3,
                           filter_parents=False, mode="graph")
        assert T.id in {r.record.id for r in graph}, (
            "graph should surface the boundary-relevant sibling mentioning Joanna"
        )
    finally:
        mem.close()


def test_graph_falls_back_when_no_entities():
    emb = {"q": [1.0, 0.0], "a": [0.9, 0.1], "b": [0.1, 0.9]}
    mem = Memory(embedding_provider=ControlledEmbedder(emb, dim=2))
    try:
        _mk(mem, "a", emb["a"]); _mk(mem, "b", emb["b"])
        res = mem.search("q", user_id="u", limit=2, filter_parents=False,
                         mode="graph")
        assert len(res) >= 1
    finally:
        mem.close()
