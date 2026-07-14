import numpy as np
import pytest

import genome
from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider


@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=16))
    yield m
    m.close()


def test_top_level_import():
    # genome.Memory works via lazy __getattr__
    assert genome.Memory is Memory


def test_add_single_identity(mem):
    recs = mem.add("user likes pour-over coffee", user_id="alice")
    assert len(recs) == 1
    assert recs[0].content == "user likes pour-over coffee"
    assert recs[0].user_id == "alice"
    assert recs[0].id.startswith("mem_")
    assert recs[0].embedding.shape == (16,)


def test_add_empty_no_records(mem):
    assert mem.add("", user_id="alice") == []
    assert mem.add("    ", user_id="alice") == []


def test_add_with_llm_extractor_multi_fact():
    def fake_llm(_prompt: str) -> str:
        return "- user likes coffee\n- user lives in Tokyo"

    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8), llm_call=fake_llm)
    try:
        recs = m.add("I love coffee and live in Tokyo", user_id="alice")
        assert len(recs) == 2
        contents = {r.content for r in recs}
        assert contents == {"user likes coffee", "user lives in Tokyo"}
    finally:
        m.close()


def test_search_returns_most_similar_first(mem):
    mem.add("user likes coffee", user_id="alice")
    mem.add("user loves pizza", user_id="alice")
    mem.add("unrelated quantum physics fact", user_id="alice")

    results = mem.search("what drinks does the user like?", user_id="alice", limit=3)
    assert len(results) == 3
    # FakeEmbeddingProvider produces hashed vectors, so we can't assert WHICH
    # is first, but all 3 memories should be returned.
    ids = {r.id for r in results}
    assert len(ids) == 3


def test_search_respects_scope(mem):
    mem.add("alice fact", user_id="alice")
    mem.add("bob fact", user_id="bob")
    alice = mem.search("fact", user_id="alice", limit=10)
    bob = mem.search("fact", user_id="bob", limit=10)
    assert len(alice) == 1
    assert len(bob) == 1
    assert alice[0].content == "alice fact"
    assert bob[0].content == "bob fact"


def test_search_limit(mem):
    for i in range(5):
        mem.add(f"fact {i}", user_id="u")
    results = mem.search("query", user_id="u", limit=2)
    assert len(results) == 2


def test_update_re_embeds(mem):
    rec = mem.add("original content", user_id="u")[0]
    before = rec.embedding.copy()
    mem.update(rec.id, content="new content")
    updated = mem.get(rec.id)
    assert updated is not None
    assert updated.content == "new content"
    # Embedding should be different (FakeEmbed is content-dependent)
    assert not np.allclose(updated.embedding, before)


def test_update_metadata_only_keeps_embedding(mem):
    rec = mem.add("hello", user_id="u")[0]
    before = rec.embedding.copy()
    mem.update(rec.id, metadata={"tag": "important"})
    updated = mem.get(rec.id)
    assert updated is not None
    assert updated.metadata == {"tag": "important"}
    np.testing.assert_allclose(updated.embedding, before)


def test_delete(mem):
    rec = mem.add("bye", user_id="u")[0]
    assert mem.delete(rec.id) is True
    assert mem.get(rec.id) is None


def test_get_touches_accessed_at(mem):
    rec = mem.add("x", user_id="u")[0]
    first = mem.get(rec.id)
    assert first is not None
    assert first.access_count == 1
    mem.get(rec.id)
    mem.get(rec.id)
    third = mem.get(rec.id)
    assert third is not None
    assert third.access_count == 4


def test_synthesize_creates_hybrid_with_provenance(mem):
    a = mem.add("fact A", user_id="alice")[0]
    b = mem.add("fact B", user_id="alice")[0]
    hybrid = mem.synthesize(
        memory_ids=[a.id, b.id], user_id="alice", operator="simple_average"
    )
    assert hybrid.is_synthesized
    assert set(hybrid.parents) == {a.id, b.id}
    assert hybrid.operator == "simple_average"
    # Embedding is average of parents
    expected = (a.embedding + b.embedding) / 2.0
    np.testing.assert_allclose(hybrid.embedding, expected, atol=1e-5)
    # Metadata carries parent contents for debuggability
    assert hybrid.metadata["parent_contents"] == ["fact A", "fact B"]


def test_synthesize_n_parents(mem):
    recs = [mem.add(f"fact {i}", user_id="u")[0] for i in range(3)]
    hybrid = mem.synthesize(
        memory_ids=[r.id for r in recs],
        user_id="u",
        operator="uniform_crossover",
        seed=42,
    )
    assert len(hybrid.parents) == 3
    # Each dim must match one of the parent values at that dim
    for i, v in enumerate(hybrid.embedding):
        parent_vals = {r.embedding[i] for r in recs}
        assert v in parent_vals


def test_synthesize_single_parent_raises(mem):
    rec = mem.add("only", user_id="u")[0]
    with pytest.raises(ValueError):
        mem.synthesize(memory_ids=[rec.id], user_id="u")


def test_synthesize_missing_parent_raises(mem):
    rec = mem.add("one", user_id="u")[0]
    with pytest.raises(ValueError):
        mem.synthesize(memory_ids=[rec.id, "mem_nonexistent"], user_id="u")


def test_search_filters_parents_by_default(mem):
    # Add atomic memories, synthesize a hybrid, search should not return the parents.
    a = mem.add("alpha concept", user_id="u")[0]
    b = mem.add("beta concept", user_id="u")[0]
    hybrid = mem.synthesize(memory_ids=[a.id, b.id], user_id="u")

    results = mem.search("a query", user_id="u", limit=10)
    returned_ids = {r.id for r in results}
    # Parents of `hybrid` should be excluded; only the hybrid remains
    assert a.id not in returned_ids
    assert b.id not in returned_ids
    assert hybrid.id in returned_ids


def test_search_filter_parents_disabled(mem):
    a = mem.add("alpha", user_id="u")[0]
    b = mem.add("beta", user_id="u")[0]
    mem.synthesize(memory_ids=[a.id, b.id], user_id="u")
    results = mem.search("q", user_id="u", filter_parents=False, limit=10)
    returned_ids = {r.id for r in results}
    assert a.id in returned_ids
    assert b.id in returned_ids


def test_count_and_reset(mem):
    for i in range(3):
        mem.add(f"fact {i}", user_id="alice")
    mem.add("other", user_id="bob")
    assert mem.count(user_id="alice") == 3
    assert mem.count(user_id="bob") == 1
    deleted = mem.reset(user_id="alice")
    assert deleted == 3
    assert mem.count(user_id="alice") == 0
    assert mem.count(user_id="bob") == 1


def test_context_manager_closes(tmp_path):
    db = tmp_path / "m.db"
    with Memory(storage=db, embedding_provider=FakeEmbeddingProvider(dim=8)) as m:
        m.add("hello", user_id="u")
        assert m.count(user_id="u") == 1

    # Reopen -- data should persist (SQLite)
    m2 = Memory(storage=db, embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        assert m2.count(user_id="u") == 1
    finally:
        m2.close()


def test_list_all_scope(mem):
    mem.add("a1", user_id="alice")
    mem.add("a2", user_id="alice")
    mem.add("b1", user_id="bob")
    assert len(mem.list_all(user_id="alice")) == 2
    assert len(mem.list_all(user_id="bob")) == 1
    assert len(mem.list_all()) == 3
