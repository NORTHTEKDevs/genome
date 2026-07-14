"""Tenant-isolation tests.

Covers the cross-tenant attack vectors fixed in v1.0.0-rc1:
- synthesize() across scopes (the CRITICAL leak)
- link() across scopes
- related() returning cross-scope records
- delete() / update() / get() without scope enforcement
"""
import pytest

from genome.errors import MemoryNotFoundError, ScopeError
from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider


@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=16))
    yield m
    m.close()


# ---------- synthesize ----------

def test_synthesize_refuses_cross_tenant_via_user_id(mem):
    """An attacker passing user_id='alice' with bob's memory_ids must be refused."""
    alice = mem.add("alice memory", user_id="alice")[0]
    bob = mem.add("bob memory", user_id="bob")[0]

    with pytest.raises(ScopeError) as ei:
        mem.synthesize(memory_ids=[alice.id, bob.id], user_id="alice")
    assert "not in user_id" in str(ei.value)


def test_synthesize_refuses_mixed_scopes_when_no_scope_given(mem):
    """If caller doesn't specify user_id, parents must still share scope."""
    alice = mem.add("alice memory", user_id="alice")[0]
    bob = mem.add("bob memory", user_id="bob")[0]

    with pytest.raises(ScopeError) as ei:
        mem.synthesize(memory_ids=[alice.id, bob.id])
    assert "multiple scopes" in str(ei.value)


def test_synthesize_refuses_cross_agent_id(mem):
    a = mem.add("a", user_id="u", agent_id="agent1")[0]
    b = mem.add("b", user_id="u", agent_id="agent2")[0]
    with pytest.raises(ScopeError):
        mem.synthesize(memory_ids=[a.id, b.id], user_id="u", agent_id="agent1")


def test_synthesize_happy_path_same_scope(mem):
    a = mem.add("alice fact 1", user_id="alice")[0]
    b = mem.add("alice fact 2", user_id="alice")[0]
    hybrid = mem.synthesize(memory_ids=[a.id, b.id], user_id="alice")
    assert hybrid.user_id == "alice"
    assert set(hybrid.parents) == {a.id, b.id}


def test_synthesize_missing_parent_still_raises(mem):
    a = mem.add("a", user_id="u")[0]
    with pytest.raises(MemoryNotFoundError):
        mem.synthesize(memory_ids=[a.id, "mem_missing"], user_id="u")


# ---------- link ----------

def test_link_refuses_cross_tenant_edge(mem):
    alice = mem.add("alice", user_id="alice")[0]
    bob = mem.add("bob", user_id="bob")[0]
    with pytest.raises(ScopeError) as ei:
        mem.link(alice.id, bob.id, relation="relates_to")
    assert "different scopes" in str(ei.value)


def test_link_refuses_cross_agent_edge(mem):
    a = mem.add("a", user_id="u", agent_id="x")[0]
    b = mem.add("b", user_id="u", agent_id="y")[0]
    with pytest.raises(ScopeError):
        mem.link(a.id, b.id, relation="relates_to")


def test_link_happy_path_same_scope(mem):
    a = mem.add("a", user_id="u")[0]
    b = mem.add("b", user_id="u")[0]
    edge = mem.link(a.id, b.id, relation="relates_to")
    assert edge.from_id == a.id
    assert edge.to_id == b.id


# ---------- related (defense in depth) ----------

def test_related_respects_user_id_filter(mem):
    """Even if the store somehow has a cross-scope edge (e.g. legacy data),
    related(user_id=...) filters it out."""
    alice = mem.add("alice", user_id="alice")[0]
    bob = mem.add("bob", user_id="bob")[0]
    # Bypass facade.link to plant a cross-scope edge directly in the store
    # (simulating data from before the link() fix).
    from genome.memory.graph import MemoryEdge
    mem.store.add_edge(
        MemoryEdge(from_id=alice.id, to_id=bob.id, relation="legacy"),
    )
    # Without user_id filter, we leak
    leaked = mem.related(alice.id, relation="legacy")
    leaked_ids = {r.id for r in leaked}
    assert bob.id in leaked_ids

    # With user_id filter, we don't
    scoped = mem.related(alice.id, relation="legacy", user_id="alice")
    scoped_ids = {r.id for r in scoped}
    assert bob.id not in scoped_ids


# ---------- delete ----------

def test_delete_refuses_cross_tenant(mem):
    alice = mem.add("alice memory", user_id="alice")[0]
    # Bob tries to delete Alice's memory
    assert mem.delete(alice.id, user_id="bob") is False
    # Alice's memory still exists
    assert mem.get(alice.id) is not None


def test_delete_succeeds_in_own_scope(mem):
    alice = mem.add("alice memory", user_id="alice")[0]
    assert mem.delete(alice.id, user_id="alice") is True
    assert mem.get(alice.id) is None


def test_delete_without_scope_still_works(mem):
    """Back-compat: delete() with no scope args still deletes (trusted caller)."""
    a = mem.add("x", user_id="u")[0]
    assert mem.delete(a.id) is True


# ---------- update ----------

def test_update_refuses_cross_tenant(mem):
    alice = mem.add("alice memory", user_id="alice")[0]
    result = mem.update(alice.id, content="hijacked", user_id="bob")
    assert result is None
    # Alice's content unchanged
    assert mem.get(alice.id).content == "alice memory"


def test_update_succeeds_in_own_scope(mem):
    alice = mem.add("alice memory", user_id="alice")[0]
    r = mem.update(alice.id, content="updated", user_id="alice")
    assert r is not None
    assert r.content == "updated"


# ---------- get ----------

def test_get_refuses_cross_tenant(mem):
    alice = mem.add("alice", user_id="alice")[0]
    # Bob's get should return None for Alice's memory
    assert mem.get(alice.id, user_id="bob") is None


def test_get_succeeds_in_own_scope(mem):
    alice = mem.add("alice", user_id="alice")[0]
    r = mem.get(alice.id, user_id="alice")
    assert r is not None
    assert r.content == "alice"


def test_get_no_scope_back_compat(mem):
    """Without scope args, get() works as before (trusted caller)."""
    r = mem.add("x", user_id="u")[0]
    back = mem.get(r.id)
    assert back is not None
