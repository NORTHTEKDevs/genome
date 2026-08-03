"""Regression tests for defects found in the 2026-08-02 stranger-readiness audit.

Each test here failed before its fix. They are grouped in one file because they share
an origin, not a module -- keeping them together makes it obvious what a future
refactor would be re-breaking.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

import pytest

from genome.memory.cache import ScopeEpochs
from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider


def _memory(**kwargs) -> Memory:
    return Memory(embedding_provider=FakeEmbeddingProvider(), **kwargs)


# ---------------------------------------------------------------------------
# consolidate() must never delete structural records
# ---------------------------------------------------------------------------


def test_consolidate_preserves_entities_and_facts():
    """Before the fix, combining the knowledge graph with consolidation silently
    destroyed the entity graph and every fact -- no error, no warning."""
    m = _memory()
    seed = m.add("Alice works at Acme Corp in Tokyo", user_id="u")[0]
    m.extract_entities(seed.id)

    entities = m.list_entities(user_id="u")
    assert entities, "precondition: entity extraction produced at least one entity"
    alice = entities[0]
    m.record_fact(alice.id, "employer", "Acme Corp")

    for i in range(25):
        m.add(f"unrelated filler memory number {i}", user_id="u")

    m.consolidate(user_id="u", max_memories=3)

    assert m.list_entities(user_id="u"), "entity records were pruned by consolidation"
    assert m.current_facts(alice.id), "entity facts were pruned by consolidation"


def test_consolidate_still_prunes_episodic_memories():
    """The protection must not turn consolidation into a no-op."""
    m = _memory()
    for i in range(20):
        m.add(f"episodic memory {i}", user_id="u")

    result = m.consolidate(user_id="u", max_memories=5)

    assert result.pruned > 0
    assert len(m.list_all()) <= 5


def test_max_memories_bounds_episodic_records_only():
    m = _memory()
    seed = m.add("Bob works at Globex", user_id="u")[0]
    m.extract_entities(seed.id)
    entity_count = len(m.list_entities(user_id="u"))
    assert entity_count > 0

    for i in range(10):
        m.add(f"filler {i}", user_id="u")

    m.consolidate(user_id="u", max_memories=2)
    assert len(m.list_entities(user_id="u")) == entity_count


# ---------------------------------------------------------------------------
# Cache epochs must isolate tenants
# ---------------------------------------------------------------------------


def test_scoped_epoch_is_unaffected_by_another_scope_mutation():
    epochs = ScopeEpochs()
    before = epochs.current("alice", None)
    epochs.bump("bob", None)
    assert epochs.current("alice", None) == before


def test_scoped_epoch_advances_on_its_own_mutation():
    epochs = ScopeEpochs()
    before = epochs.current("alice", None)
    epochs.bump("alice", None)
    assert epochs.current("alice", None) != before


def test_unscoped_epoch_advances_on_any_mutation():
    """An unscoped query spans every scope, so it must invalidate on any write."""
    epochs = ScopeEpochs()
    before = epochs.current(None, None)
    epochs.bump("bob", None)
    assert epochs.current(None, None) != before


def test_reset_all_invalidates_without_moving_epochs_backwards():
    """A decreasing counter could let a stale entry be re-hit once a scope caught
    back up."""
    epochs = ScopeEpochs()
    epochs.bump("alice", None)
    epochs.bump("alice", None)
    seen = epochs.current("alice", None)

    epochs.reset_all()
    after_reset = epochs.current("alice", None)
    assert after_reset != seen

    epochs.bump("alice", None)
    epochs.bump("alice", None)
    assert epochs.current("alice", None) != seen


def test_one_tenants_write_does_not_evict_another_tenants_cached_search():
    """The end-to-end symptom: multi-tenant hit rate collapsing to zero."""
    m = _memory(enable_cache=True)
    m.add("alice likes pour-over coffee", user_id="alice")
    m.add("bob likes tea", user_id="bob")

    m.search("coffee", user_id="alice")
    m.search("coffee", user_id="alice")
    hits_before = m.cache_stats.hits
    assert hits_before >= 1, "precondition: the repeated query was served from cache"

    m.add("bob also likes biscuits", user_id="bob")
    m.search("coffee", user_id="alice")

    assert m.cache_stats.hits == hits_before + 1, (
        "a write to bob's scope invalidated alice's cached query"
    )


# ---------------------------------------------------------------------------
# Input validation at the public boundary
# ---------------------------------------------------------------------------


def test_negative_search_limit_is_refused_not_silently_truncated():
    """Python slice semantics turned limit=-1 into 'all but the last result'."""
    m = _memory()
    for i in range(3):
        m.add(f"corpus item {i}", user_id="u")

    with pytest.raises(ValueError, match="limit must be >= 0"):
        m.search("corpus", user_id="u", limit=-1)


def test_zero_search_limit_returns_nothing():
    m = _memory()
    m.add("corpus item", user_id="u")
    assert m.search("corpus", user_id="u", limit=0) == []


def test_non_integer_search_limit_is_refused():
    m = _memory()
    m.add("corpus item", user_id="u")
    with pytest.raises(TypeError, match="limit must be an int"):
        m.search("corpus", user_id="u", limit="5")


@pytest.mark.parametrize("bad_text", [None, 12345, ["a", "b"], {"a": 1}])
def test_add_rejects_non_string_text_with_an_actionable_error(bad_text):
    m = _memory()
    with pytest.raises(TypeError, match="text must be a string"):
        m.add(bad_text, user_id="u")


def test_add_rejects_integer_user_id_with_a_hint():
    """Integer primary keys are the most common user_id representation there is."""
    m = _memory()
    with pytest.raises(TypeError, match="user_id must be a string"):
        m.add("hello there", user_id=12345)


def test_add_rejects_integer_agent_id():
    m = _memory()
    with pytest.raises(TypeError, match="agent_id must be a string"):
        m.add("hello there", agent_id=999)


def test_add_rejects_non_dict_metadata():
    m = _memory()
    with pytest.raises(TypeError, match="metadata must be a dict"):
        m.add("hello there", user_id="u", metadata='{"pre": "serialized"}')


def test_valid_add_still_works():
    """Guard rails must not block the happy path."""
    m = _memory()
    records = m.add("a perfectly ordinary memory", user_id="u", metadata={"k": "v"})
    assert len(records) >= 1
    assert m.search("ordinary", user_id="u", limit=5)
