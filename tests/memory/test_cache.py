import threading

import pytest

from genome.memory.cache import CacheStats, ResponseCache, ScopeEpochs
from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider

# ---------- unit tests for ResponseCache ----------

def test_cache_basic_put_get():
    c = ResponseCache(capacity=10)
    assert c.get("q", "u", None, 5, True, 1) is None
    c.put("q", "u", None, 5, True, 1, ["r1", "r2"])
    assert c.get("q", "u", None, 5, True, 1) == ["r1", "r2"]


def test_cache_miss_increments_stats():
    c = ResponseCache()
    c.get("q", "u", None, 5, True, 0)
    assert c.stats.misses == 1
    assert c.stats.hits == 0
    c.put("q", "u", None, 5, True, 0, ["r"])
    c.get("q", "u", None, 5, True, 0)
    assert c.stats.hits == 1


def test_cache_hit_rate():
    c = ResponseCache()
    c.get("q", None, None, 5, True, 0)  # miss
    c.put("q", None, None, 5, True, 0, [])
    c.get("q", None, None, 5, True, 0)  # hit
    c.get("q", None, None, 5, True, 0)  # hit
    assert c.stats.hit_rate == pytest.approx(2 / 3)


def test_cache_different_epochs_isolate():
    c = ResponseCache()
    c.put("q", "u", None, 5, True, 1, ["a"])
    c.put("q", "u", None, 5, True, 2, ["b"])
    assert c.get("q", "u", None, 5, True, 1) == ["a"]
    assert c.get("q", "u", None, 5, True, 2) == ["b"]


def test_cache_capacity_evicts_lru():
    c = ResponseCache(capacity=2)
    c.put("q1", None, None, 5, True, 0, [1])
    c.put("q2", None, None, 5, True, 0, [2])
    c.put("q3", None, None, 5, True, 0, [3])
    assert c.get("q1", None, None, 5, True, 0) is None
    assert c.get("q2", None, None, 5, True, 0) == [2]
    assert c.get("q3", None, None, 5, True, 0) == [3]


def test_cache_clear_resets():
    c = ResponseCache()
    c.put("q", None, None, 5, True, 0, [1])
    c.clear()
    assert len(c) == 0
    assert c.get("q", None, None, 5, True, 0) is None


def test_cache_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ResponseCache(capacity=0)


def test_cache_stats_default():
    s = CacheStats()
    assert s.hits == 0
    assert s.misses == 0
    assert s.hit_rate == 0.0


# ---------- thread safety (fixed in v1.0.0-rc1 R3) ----------

def test_cache_thread_safe_under_concurrent_mutation():
    """100 threads doing interleaved put/get must not crash or corrupt state."""
    c = ResponseCache(capacity=100)

    errors: list[Exception] = []

    def worker(seed: int):
        try:
            for i in range(50):
                c.put(f"q{seed}-{i}", None, None, 5, True, 0, [seed, i])
                _ = c.get(f"q{seed}-{i}", None, None, 5, True, 0)
        except Exception as e:  # pragma: no cover - shouldn't happen
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"thread errors: {errors[:3]}"
    # Cache still functional post-concurrent hammer
    c.put("final", None, None, 5, True, 0, ["ok"])
    assert c.get("final", None, None, 5, True, 0) == ["ok"]


# ---------- ScopeEpochs ----------

def test_scope_epochs_bump_invalidates_scope():
    se = ScopeEpochs()
    e1 = se.current("alice", None)
    se.bump("alice", None)
    e2 = se.current("alice", None)
    assert e2 != e1


def test_scope_epochs_bump_of_one_scope_affects_others_via_global():
    """Unscoped queries must invalidate when any scope mutates (global epoch)."""
    se = ScopeEpochs()
    global_before = se.current(None, None)
    se.bump("alice", None)
    global_after = se.current(None, None)
    assert global_after != global_before


def test_scope_epochs_thread_safe():
    se = ScopeEpochs()
    errors: list[Exception] = []

    def worker(user: str):
        try:
            for _ in range(1000):
                se.bump(user, None)
                _ = se.current(user, None)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(f"u{i}",)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


# ---------- Integration with Memory facade ----------

@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=16))
    yield m
    m.close()


def test_memory_search_caches(mem):
    mem.add("fact 1", user_id="u")
    mem.add("fact 2", user_id="u")

    r1 = mem.search("query", user_id="u")
    stats = mem.cache_stats
    assert stats is not None
    assert stats.misses == 1
    assert stats.hits == 0

    r2 = mem.search("query", user_id="u")
    stats2 = mem.cache_stats
    assert stats2.hits == 1
    assert stats2.misses == 1
    assert [x.id for x in r1] == [x.id for x in r2]


def test_memory_cache_invalidates_on_add(mem):
    mem.add("fact 1", user_id="u")
    mem.search("q", user_id="u")
    assert mem.cache_stats.misses == 1
    mem.add("fact 2", user_id="u")
    mem.search("q", user_id="u")
    assert mem.cache_stats.misses == 2


def test_memory_cache_invalidates_on_delete(mem):
    r1 = mem.add("fact 1", user_id="u")[0]
    mem.add("fact 2", user_id="u")
    mem.search("q", user_id="u")
    misses_before = mem.cache_stats.misses
    mem.delete(r1.id)
    mem.search("q", user_id="u")
    assert mem.cache_stats.misses == misses_before + 1


def test_memory_cache_invalidates_on_explicit_consolidate(mem):
    """An explicit consolidate() must invalidate cached search results for the
    scope -- otherwise a search after pruning serves just-deleted records."""
    for i in range(6):
        mem.add(f"fact {i}", user_id="u")
    mem.search("q", user_id="u")
    misses_before = mem.cache_stats.misses
    # Prune hard so consolidation actually mutates the scope.
    mem.consolidate(user_id="u", max_memories=2)
    mem.search("q", user_id="u")
    assert mem.cache_stats.misses == misses_before + 1


def test_memory_cache_invalidates_on_reset(mem):
    mem.add("x", user_id="u")
    mem.search("q", user_id="u")
    misses_before = mem.cache_stats.misses
    mem.reset(user_id="u")
    mem.search("q", user_id="u")
    assert mem.cache_stats.misses == misses_before + 1


def test_memory_cache_scoped_by_user(mem):
    mem.add("alice fact", user_id="alice")
    mem.add("bob fact", user_id="bob")
    mem.search("q", user_id="alice")
    mem.search("q", user_id="bob")
    mem.search("q", user_id="alice")
    mem.search("q", user_id="bob")
    assert mem.cache_stats.hits == 2
    assert mem.cache_stats.misses == 2


def test_memory_cache_respects_limit_param(mem):
    mem.add("f1", user_id="u")
    mem.add("f2", user_id="u")
    mem.search("q", user_id="u", limit=1)
    mem.search("q", user_id="u", limit=3)
    mem.search("q", user_id="u", limit=1)
    assert mem.cache_stats.hits == 1
    assert mem.cache_stats.misses == 2


def test_memory_cache_use_cache_false(mem):
    mem.add("fact", user_id="u")
    mem.search("q", user_id="u")
    mem.search("q", user_id="u", use_cache=False)
    assert mem.cache_stats.misses == 1


def test_memory_disable_cache_constructor():
    m = Memory(
        embedding_provider=FakeEmbeddingProvider(dim=8),
        enable_cache=False,
    )
    try:
        m.add("x", user_id="u")
        m.search("q", user_id="u")
        m.search("q", user_id="u")
        assert m.cache_stats is None
    finally:
        m.close()


def test_memory_manual_clear_cache(mem):
    mem.add("f", user_id="u")
    mem.search("q", user_id="u")
    mem.search("q", user_id="u")
    assert mem.cache_stats.hits == 1
    mem.clear_cache()
    mem.search("q", user_id="u")
    assert mem.cache_stats.hits == 1
    assert mem.cache_stats.misses == 2


# ---------- Cache invalidation on graph + synthesis (new in R3) ----------

def test_cache_invalidates_on_synthesize(mem):
    a = mem.add("alpha", user_id="u")[0]
    b = mem.add("beta", user_id="u")[0]
    mem.search("q", user_id="u")  # populate
    misses_before = mem.cache_stats.misses
    mem.synthesize(memory_ids=[a.id, b.id], user_id="u", operator="simple_average")
    mem.search("q", user_id="u")
    assert mem.cache_stats.misses == misses_before + 1


def test_cache_invalidates_on_link(mem):
    a = mem.add("a", user_id="u")[0]
    b = mem.add("b", user_id="u")[0]
    mem.search("q", user_id="u")
    misses_before = mem.cache_stats.misses
    mem.link(a.id, b.id, relation="relates_to")
    mem.search("q", user_id="u")
    assert mem.cache_stats.misses == misses_before + 1


def test_cache_invalidates_on_unlink(mem):
    a = mem.add("a", user_id="u")[0]
    b = mem.add("b", user_id="u")[0]
    edge = mem.link(a.id, b.id, relation="relates_to")
    mem.search("q", user_id="u")
    misses_before = mem.cache_stats.misses
    mem.unlink(edge.id)
    mem.search("q", user_id="u")
    assert mem.cache_stats.misses == misses_before + 1


# ---------- No O(n) scope scan on cache miss (R3 fix) ----------

def test_cache_miss_is_constant_time():
    """Adding 500 memories should not make cache miss 500x slower.

    We can't measure nanoseconds reliably, but we can confirm the cache path
    does not call list_by_scope (the old O(n) fingerprint).
    """
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        for i in range(500):
            m.add(f"fact {i}", user_id="u")

        # Monkey-patch list_by_scope to count calls
        calls = []
        orig = m.store.list_by_scope

        def counting_list_by_scope(*args, **kw):
            calls.append(1)
            return orig(*args, **kw)

        m.store.list_by_scope = counting_list_by_scope  # type: ignore[method-assign]

        calls.clear()
        m.search("q", user_id="u")  # cache miss
        # list_by_scope is called ONCE (for the filter_parents pass),
        # not twice (as it would be if scope_fingerprint was still there).
        assert len(calls) == 1
    finally:
        m.close()
