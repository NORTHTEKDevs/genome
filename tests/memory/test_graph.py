import pytest

from genome.memory.facade import Memory
from genome.memory.graph import (
    CAUSES,
    CONTRADICTS,
    DERIVED_FROM,
    RELATES_TO,
    SUPERSEDES,
    MemoryEdge,
)
from tests.memory._fake_embed import FakeEmbeddingProvider


@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    yield m
    m.close()


def test_edge_validation():
    import numpy as np
    # valid
    e = MemoryEdge(from_id="a", to_id="b", relation="rel", weight=0.7)
    assert e.id.startswith("edge_")
    assert e.weight == 0.7
    # empty endpoints
    with pytest.raises(ValueError):
        MemoryEdge(from_id="", to_id="b", relation="x")
    # empty relation
    with pytest.raises(ValueError):
        MemoryEdge(from_id="a", to_id="b", relation="")
    # weight out of range
    with pytest.raises(ValueError):
        MemoryEdge(from_id="a", to_id="b", relation="r", weight=1.5)
    with pytest.raises(ValueError):
        MemoryEdge(from_id="a", to_id="b", relation="r", weight=-0.1)
    # pyflakes
    _ = np


def test_link_and_related(mem):
    a = mem.add("fact A", user_id="u")[0]
    b = mem.add("fact B", user_id="u")[0]
    edge = mem.link(a.id, b.id, relation=SUPERSEDES)
    assert edge.from_id == a.id
    assert edge.to_id == b.id
    assert edge.relation == SUPERSEDES

    # outgoing from A = [B]
    out = mem.related(a.id, relation=SUPERSEDES, direction="out")
    assert len(out) == 1
    assert out[0].id == b.id

    # incoming to B = [A]
    inc = mem.related(b.id, relation=SUPERSEDES, direction="in")
    assert len(inc) == 1
    assert inc[0].id == a.id


def test_related_filters_by_relation(mem):
    a = mem.add("a", user_id="u")[0]
    b = mem.add("b", user_id="u")[0]
    c = mem.add("c", user_id="u")[0]
    mem.link(a.id, b.id, relation=SUPERSEDES)
    mem.link(a.id, c.id, relation=CONTRADICTS)

    supersedes_targets = mem.related(a.id, relation=SUPERSEDES)
    assert {r.id for r in supersedes_targets} == {b.id}

    contradicts_targets = mem.related(a.id, relation=CONTRADICTS)
    assert {r.id for r in contradicts_targets} == {c.id}

    all_targets = mem.related(a.id)
    assert {r.id for r in all_targets} == {b.id, c.id}


def test_related_direction_both(mem):
    a = mem.add("a", user_id="u")[0]
    b = mem.add("b", user_id="u")[0]
    c = mem.add("c", user_id="u")[0]
    mem.link(a.id, b.id, relation=RELATES_TO)
    mem.link(c.id, a.id, relation=RELATES_TO)

    out = mem.related(a.id, relation=RELATES_TO, direction="out")
    inc = mem.related(a.id, relation=RELATES_TO, direction="in")
    both = mem.related(a.id, relation=RELATES_TO, direction="both")
    assert {r.id for r in out} == {b.id}
    assert {r.id for r in inc} == {c.id}
    assert {r.id for r in both} == {b.id, c.id}


def test_link_missing_endpoints_raises(mem):
    a = mem.add("a", user_id="u")[0]
    with pytest.raises(ValueError):
        mem.link(a.id, "mem_nonexistent", relation=RELATES_TO)
    with pytest.raises(ValueError):
        mem.link("mem_nonexistent", a.id, relation=RELATES_TO)


def test_unlink(mem):
    a = mem.add("a", user_id="u")[0]
    b = mem.add("b", user_id="u")[0]
    edge = mem.link(a.id, b.id, relation=DERIVED_FROM)
    assert mem.unlink(edge.id) is True
    # Second unlink returns False
    assert mem.unlink(edge.id) is False
    assert mem.related(a.id, relation=DERIVED_FROM) == []


def test_delete_memory_cascades_edges(mem):
    a = mem.add("a", user_id="u")[0]
    b = mem.add("b", user_id="u")[0]
    c = mem.add("c", user_id="u")[0]
    mem.link(a.id, b.id, relation=CAUSES)
    mem.link(b.id, c.id, relation=CAUSES)

    # Delete b - both edges (a->b and b->c) should go
    mem.delete(b.id)
    assert mem.related(a.id, relation=CAUSES) == []
    assert mem.related(c.id, relation=CAUSES, direction="in") == []


def test_edges_of(mem):
    a = mem.add("a", user_id="u")[0]
    b = mem.add("b", user_id="u")[0]
    mem.link(a.id, b.id, relation=SUPERSEDES, weight=0.8, metadata={"reason": "new"})
    edges = mem.edges_of(a.id, relation=SUPERSEDES)
    assert len(edges) == 1
    assert edges[0].weight == 0.8
    assert edges[0].metadata == {"reason": "new"}


def test_common_relation_constants_are_unique():
    constants = {SUPERSEDES, CONTRADICTS, DERIVED_FROM, RELATES_TO, CAUSES}
    assert len(constants) == 5


def test_unlink_refuses_cross_scope(mem):
    a = mem.add("a", user_id="alice")[0]
    b = mem.add("b", user_id="alice")[0]
    edge = mem.link(a.id, b.id, relation=RELATES_TO)
    # bob can't delete alice's edge by guessing the id
    assert mem.unlink(edge.id, user_id="bob") is False
    # alice still can
    assert mem.unlink(edge.id, user_id="alice") is True


def test_edges_of_refuses_cross_scope(mem):
    a = mem.add("a", user_id="alice")[0]
    b = mem.add("b", user_id="alice")[0]
    mem.link(a.id, b.id, relation=SUPERSEDES)
    # bob can't enumerate alice's edges by guessing the anchor id
    assert mem.edges_of(a.id, user_id="bob") == []
    # alice can
    assert len(mem.edges_of(a.id, user_id="alice")) == 1
