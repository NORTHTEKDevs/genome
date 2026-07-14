import numpy as np
import pytest

from genome.memory.schema import MemoryRecord
from genome.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def store():
    s = SQLiteMemoryStore(":memory:")
    yield s
    s.close()


def _rec(content: str, dim: int = 8, **kw) -> MemoryRecord:
    rng = np.random.default_rng(abs(hash(content)) % (2**32))
    return MemoryRecord(
        content=content,
        embedding=rng.standard_normal(dim).astype(np.float32),
        **kw,
    )


def test_add_and_get(store):
    r = _rec("hello world")
    stored = store.add(r)
    assert stored.id == r.id
    back = store.get(r.id)
    assert back is not None
    assert back.content == "hello world"
    assert back.embedding.shape == (8,)
    np.testing.assert_allclose(back.embedding, r.embedding, atol=1e-6)


def test_get_missing_returns_none(store):
    assert store.get("mem_nonexistent") is None


def test_update_content(store):
    r = store.add(_rec("original"))
    out = store.update(r.id, content="updated")
    assert out is not None
    assert out.content == "updated"
    # Embedding unchanged
    np.testing.assert_allclose(out.embedding, r.embedding, atol=1e-6)


def test_update_embedding(store):
    r = store.add(_rec("x"))
    new_emb = np.ones(8, dtype=np.float32)
    out = store.update(r.id, embedding=new_emb)
    assert out is not None
    np.testing.assert_allclose(out.embedding, new_emb)


def test_update_missing(store):
    assert store.update("mem_nope", content="x") is None


def test_delete(store):
    r = store.add(_rec("bye"))
    assert store.delete(r.id) is True
    assert store.get(r.id) is None
    assert store.delete(r.id) is False


def test_search_scoped_by_user(store):
    # Alice and Bob each have a memory. Search in Alice's scope only.
    emb_alice = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    emb_bob = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    store.add(MemoryRecord(content="alice_mem", embedding=emb_alice, user_id="alice"))
    store.add(MemoryRecord(content="bob_mem", embedding=emb_bob, user_id="bob"))
    q = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    alice_results = store.search(q, user_id="alice")
    assert len(alice_results) == 1
    assert alice_results[0].content == "alice_mem"

    bob_results = store.search(q, user_id="bob")
    assert len(bob_results) == 1
    assert bob_results[0].content == "bob_mem"


def test_search_exclude_ids(store):
    emb = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    r1 = store.add(MemoryRecord(content="a", embedding=emb, user_id="u"))
    r2 = store.add(MemoryRecord(content="b", embedding=emb, user_id="u"))
    results = store.search(emb, user_id="u", exclude_ids={r1.id})
    assert len(results) == 1
    assert results[0].id == r2.id


def test_search_limits(store):
    for i in range(5):
        emb = np.zeros(8, dtype=np.float32)
        emb[i] = 1.0
        store.add(MemoryRecord(content=f"m{i}", embedding=emb, user_id="u"))
    q = np.ones(8, dtype=np.float32)
    results = store.search(q, user_id="u", limit=3)
    assert len(results) == 3


def test_list_by_scope(store):
    store.add(MemoryRecord(content="a", embedding=np.zeros(4, dtype=np.float32), user_id="u", agent_id="x"))
    store.add(MemoryRecord(content="b", embedding=np.zeros(4, dtype=np.float32), user_id="u", agent_id="y"))
    assert len(store.list_by_scope(user_id="u")) == 2
    assert len(store.list_by_scope(user_id="u", agent_id="x")) == 1


def test_count(store):
    for i in range(3):
        store.add(MemoryRecord(content=f"m{i}", embedding=np.zeros(4, dtype=np.float32), user_id="u"))
    assert store.count(user_id="u") == 3
    assert store.count(user_id="other") == 0


def test_touch_increments_access(store):
    r = store.add(_rec("x"))
    assert r.access_count == 0
    store.touch(r.id)
    store.touch(r.id)
    back = store.get(r.id)
    assert back is not None
    assert back.access_count == 2
    assert back.accessed_at >= r.accessed_at


def test_persistence_across_connections(tmp_path):
    db = tmp_path / "mem.db"
    s1 = SQLiteMemoryStore(db)
    r = s1.add(MemoryRecord(
        content="persistent",
        embedding=np.ones(4, dtype=np.float32),
        user_id="u",
    ))
    s1.close()

    s2 = SQLiteMemoryStore(db)
    back = s2.get(r.id)
    assert back is not None
    assert back.content == "persistent"
    s2.close()


def test_synthesized_memory_roundtrip(store):
    r = store.add(MemoryRecord(
        content="hybrid mem",
        embedding=np.zeros(4, dtype=np.float32),
        parents=["mem_a", "mem_b"],
        operator="uniform_crossover",
        metadata={"note": "from synthesis"},
    ))
    back = store.get(r.id)
    assert back is not None
    assert back.is_synthesized
    assert back.parents == ["mem_a", "mem_b"]
    assert back.operator == "uniform_crossover"
    assert back.metadata == {"note": "from synthesis"}
