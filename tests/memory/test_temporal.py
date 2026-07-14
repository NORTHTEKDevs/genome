"""Tests for the temporal knowledge graph (Zep-parity feature)."""
import time

import pytest

from genome.errors import MemoryNotFoundError, ScopeError
from genome.memory.entities import ENTITY_OPERATOR
from genome.memory.facade import Memory
from genome.memory.schema import MemoryRecord
from genome.memory.temporal import (
    FACT_OPERATOR,
    current_facts,
    entity_timeline,
    facts_valid_at,
    invalidate_fact,
    merge_entity_facts,
    record_fact,
)
from tests.memory._fake_embed import FakeEmbeddingProvider


@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=16))
    yield m
    m.close()


def _make_entity(mem, name: str, user_id: str = "u") -> MemoryRecord:
    """Directly create an entity record (bypassing LLM extraction)."""
    import numpy as np
    rec = MemoryRecord(
        content=name,
        embedding=np.asarray(mem.embed.encode(name), dtype=np.float32),
        user_id=user_id,
        operator=ENTITY_OPERATOR,
        metadata={"entity_type": "PERSON", "entity_name": name},
    )
    mem.store.add(rec)
    return rec


def test_current_facts_excludes_future_dated_open_fact(mem):
    """A fact whose valid_from is in the future must NOT be reported as
    current now, even though its valid_until is None."""
    alice = _make_entity(mem, "Alice")
    future = time.time() + 365 * 24 * 3600
    record_fact(mem, alice.id, "title", "Future CEO", valid_from=future)
    currents = current_facts(mem, alice.id)
    assert all(f.value != "Future CEO" for f in currents), (
        f"future-dated fact leaked into current_facts: {[f.value for f in currents]}"
    )


def test_backdating_does_not_leave_two_current_facts(mem):
    """Recording a backdated fact ahead of an existing future-dated open fact
    must not produce two simultaneously-current facts for the same slot."""
    alice = _make_entity(mem, "Alice")
    future = time.time() + 365 * 24 * 3600
    # F1: future-dated, open.
    record_fact(mem, alice.id, "lives_in", "Tokyo", valid_from=future)
    # F2: backdated to now (earlier than F1), default invalidate_previous=True.
    record_fact(mem, alice.id, "lives_in", "NYC", valid_from=time.time())
    currents = current_facts(mem, alice.id)
    lives_in_now = [f for f in currents if f.fact_type == "lives_in"]
    # Exactly one 'lives_in' fact is true right now (NYC), not two.
    assert len(lives_in_now) == 1, (
        f"expected 1 current lives_in, got {[f.value for f in lives_in_now]}"
    )
    assert lives_in_now[0].value == "NYC"


# ---------- record_fact ----------

def test_record_fact_creates_current_fact(mem):
    alice = _make_entity(mem, "Alice")
    fact = record_fact(mem, alice.id, "lives_in", "NYC")
    assert fact.entity_id == alice.id
    assert fact.fact_type == "lives_in"
    assert fact.value == "NYC"
    assert fact.is_current
    assert fact.valid_until is None
    assert fact.valid_from <= time.time()


def test_record_fact_stores_as_memory_record(mem):
    alice = _make_entity(mem, "Alice")
    fact = record_fact(mem, alice.id, "works_at", "OpenAI")
    rec = mem.store.get(fact.id)
    assert rec is not None
    assert rec.operator == FACT_OPERATOR
    assert rec.metadata["entity_id"] == alice.id
    assert rec.content == "works_at: OpenAI"


def test_record_fact_missing_entity_raises(mem):
    with pytest.raises(MemoryNotFoundError):
        record_fact(mem, "mem_nonexistent", "lives_in", "Mars")


def test_record_fact_rejects_non_entity(mem):
    # A plain memory, not an entity record
    plain = mem.add("hello", user_id="u")[0]
    with pytest.raises(ValueError, match="entity record"):
        record_fact(mem, plain.id, "anything", "value")


def test_record_fact_invalidates_previous_current_fact(mem):
    alice = _make_entity(mem, "Alice")
    # First state: lives in NYC
    old = record_fact(mem, alice.id, "lives_in", "NYC")
    time.sleep(0.01)  # ensure distinguishable timestamps
    # New state: lives in Tokyo
    new = record_fact(mem, alice.id, "lives_in", "Tokyo")

    # Old fact should now be closed
    timeline = entity_timeline(mem, alice.id)
    by_id = {f.id: f for f in timeline}
    assert by_id[old.id].valid_until is not None
    # New fact is current
    assert by_id[new.id].valid_until is None


def test_record_fact_invalidate_previous_false_keeps_both_open(mem):
    alice = _make_entity(mem, "Alice")
    record_fact(mem, alice.id, "hobby", "chess")
    # second hobby recorded with invalidate_previous=False -> both current
    record_fact(mem, alice.id, "hobby", "golf", invalidate_previous=False)
    currents = current_facts(mem, alice.id)
    hobbies = {f.value for f in currents if f.fact_type == "hobby"}
    assert hobbies == {"chess", "golf"}


def test_record_fact_different_types_coexist(mem):
    alice = _make_entity(mem, "Alice")
    record_fact(mem, alice.id, "lives_in", "NYC")
    record_fact(mem, alice.id, "works_at", "OpenAI")
    currents = current_facts(mem, alice.id)
    types = {f.fact_type for f in currents}
    assert types == {"lives_in", "works_at"}


def test_record_fact_with_explicit_valid_from(mem):
    alice = _make_entity(mem, "Alice")
    t0 = time.time() - 86400  # yesterday
    fact = record_fact(mem, alice.id, "age", "34", valid_from=t0)
    assert fact.valid_from == pytest.approx(t0, abs=1e-3)


def test_record_fact_with_confidence(mem):
    alice = _make_entity(mem, "Alice")
    fact = record_fact(mem, alice.id, "age", "34", confidence=0.7)
    assert fact.confidence == 0.7


def test_record_fact_with_source_memory(mem):
    alice = _make_entity(mem, "Alice")
    src = mem.add("Alice is 34 years old", user_id="u")[0]
    fact = record_fact(mem, alice.id, "age", "34", source_memory_id=src.id)
    assert fact.source_memory_id == src.id


# ---------- invalidate_fact ----------

def test_invalidate_fact_sets_valid_until(mem):
    alice = _make_entity(mem, "Alice")
    fact = record_fact(mem, alice.id, "status", "active")
    assert fact.is_current
    updated = invalidate_fact(mem, fact.id)
    assert updated is not None
    assert not updated.is_current
    assert updated.valid_until is not None


def test_invalidate_fact_custom_time(mem):
    alice = _make_entity(mem, "Alice")
    fact = record_fact(mem, alice.id, "title", "Engineer")
    t = time.time() - 3600
    updated = invalidate_fact(mem, fact.id, at=t)
    assert updated.valid_until == pytest.approx(t, abs=1e-3)


def test_invalidate_missing_fact_returns_none(mem):
    assert invalidate_fact(mem, "mem_nope") is None


def test_invalidate_non_fact_raises(mem):
    plain = mem.add("not a fact", user_id="u")[0]
    with pytest.raises(ValueError, match="entity_fact"):
        invalidate_fact(mem, plain.id)


# ---------- entity_timeline ----------

def test_entity_timeline_returns_newest_first(mem):
    alice = _make_entity(mem, "Alice")
    t1 = time.time() - 100
    t2 = time.time() - 50
    t3 = time.time()
    record_fact(mem, alice.id, "role", "IC", valid_from=t1, invalidate_previous=False)
    record_fact(mem, alice.id, "role", "Senior", valid_from=t2, invalidate_previous=False)
    record_fact(mem, alice.id, "role", "Staff", valid_from=t3, invalidate_previous=False)

    timeline = entity_timeline(mem, alice.id)
    roles = [f.value for f in timeline if f.fact_type == "role"]
    assert roles == ["Staff", "Senior", "IC"]


def test_entity_timeline_respects_user_id_scope(mem):
    alice = _make_entity(mem, "Alice", user_id="u1")
    _make_entity(mem, "Alice", user_id="u2")  # noqa: different entity, same name
    record_fact(mem, alice.id, "lives_in", "NYC")

    # Wrong scope -> empty
    assert entity_timeline(mem, alice.id, user_id="u2") == []
    # Right scope -> non-empty
    assert len(entity_timeline(mem, alice.id, user_id="u1")) > 0


def test_entity_timeline_missing_entity_empty(mem):
    assert entity_timeline(mem, "mem_nope") == []


# ---------- current_facts ----------

def test_current_facts_only_returns_valid(mem):
    alice = _make_entity(mem, "Alice")
    record_fact(mem, alice.id, "city", "NYC")
    record_fact(mem, alice.id, "city", "Tokyo")  # invalidates NYC
    record_fact(mem, alice.id, "status", "active")

    currents = current_facts(mem, alice.id)
    current_values = {(f.fact_type, f.value) for f in currents}
    assert ("city", "Tokyo") in current_values
    assert ("city", "NYC") not in current_values
    assert ("status", "active") in current_values


# ---------- facts_valid_at ----------

def test_facts_valid_at_returns_historical_state(mem):
    alice = _make_entity(mem, "Alice")
    t_nyc = time.time() - 200
    t_tokyo = time.time() - 100
    record_fact(mem, alice.id, "city", "NYC", valid_from=t_nyc)
    record_fact(mem, alice.id, "city", "Tokyo", valid_from=t_tokyo)

    # At t_nyc + 50 (before Tokyo): Alice was in NYC
    valid_at_middle = facts_valid_at(mem, alice.id, t_nyc + 50)
    city_at_middle = [f for f in valid_at_middle if f.fact_type == "city"]
    assert len(city_at_middle) == 1
    assert city_at_middle[0].value == "NYC"

    # Now: Alice is in Tokyo
    current = facts_valid_at(mem, alice.id, time.time())
    city_now = [f for f in current if f.fact_type == "city"]
    assert len(city_now) == 1
    assert city_now[0].value == "Tokyo"


def test_facts_valid_at_before_any_fact_returns_empty(mem):
    alice = _make_entity(mem, "Alice")
    record_fact(mem, alice.id, "role", "Eng")
    # Query way in the past
    assert facts_valid_at(mem, alice.id, time.time() - 99999) == []


# ---------- merge_entity_facts ----------

def test_merge_entity_facts_moves_all(mem):
    alice_v1 = _make_entity(mem, "Alice Smith")
    alice_v2 = _make_entity(mem, "Alice S.")
    record_fact(mem, alice_v1.id, "lives_in", "NYC")
    record_fact(mem, alice_v1.id, "works_at", "OpenAI")

    moved = merge_entity_facts(mem, alice_v1.id, alice_v2.id)
    assert moved == 2

    # All facts now belong to alice_v2
    timeline_v2 = entity_timeline(mem, alice_v2.id)
    assert len(timeline_v2) == 2
    # alice_v1 has no more facts
    assert entity_timeline(mem, alice_v1.id) == []


def test_merge_entity_facts_cross_scope_refused(mem):
    a = _make_entity(mem, "Alice", user_id="u1")
    b = _make_entity(mem, "Alice", user_id="u2")
    with pytest.raises(ScopeError):
        merge_entity_facts(mem, a.id, b.id)


def test_merge_missing_source_raises(mem):
    b = _make_entity(mem, "B")
    with pytest.raises(MemoryNotFoundError):
        merge_entity_facts(mem, "mem_nope", b.id)


# ---------- Memory facade shortcuts ----------

def test_facade_shortcuts_work(mem):
    alice = _make_entity(mem, "Alice")
    # Use explicit timestamps to avoid sub-microsecond ordering flakiness
    t_nyc = time.time() - 100
    t_tokyo = time.time() - 50
    f = mem.record_fact(alice.id, "lives_in", "NYC", valid_from=t_nyc)
    assert f.is_current

    timeline = mem.entity_timeline(alice.id)
    assert len(timeline) == 1

    currents = mem.current_facts(alice.id)
    assert len(currents) == 1

    mem.record_fact(alice.id, "lives_in", "Tokyo", valid_from=t_tokyo)
    assert len(mem.current_facts(alice.id)) == 1
    assert mem.current_facts(alice.id)[0].value == "Tokyo"

    # Historical query at t_nyc + 10 (before t_tokyo): Alice was in NYC
    past = mem.facts_valid_at(alice.id, t_nyc + 10)
    assert any(fact.value == "NYC" for fact in past)


def test_record_fact_concurrent_no_double_open_facts(mem):
    """Two threads recording facts for the same (entity, fact_type) MUST
    end up with at most ONE valid_until=None fact for that slot. Without
    the _fact_mutation_lock, both threads can find the same prior
    (valid_until=None), both close it, both add a new fact, leaving two
    simultaneously-valid facts and a corrupt timeline."""
    import threading
    alice = _make_entity(mem, "Alice")
    # Pre-populate one current fact so both contenders see the same prior
    mem.record_fact(alice.id, "lives_in", "Original")

    barrier = threading.Barrier(4)

    def writer(value):
        barrier.wait()
        mem.record_fact(alice.id, "lives_in", value)

    threads = [
        threading.Thread(target=writer, args=("NYC",)),
        threading.Thread(target=writer, args=("Tokyo",)),
        threading.Thread(target=writer, args=("Berlin",)),
        threading.Thread(target=writer, args=("Paris",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Invariant: exactly ONE current fact for (alice, lives_in).
    currents = mem.current_facts(alice.id)
    lives_in_currents = [f for f in currents if f.fact_type == "lives_in"]
    assert len(lives_in_currents) == 1, (
        f"timeline corruption: {len(lives_in_currents)} simultaneously-valid "
        f"lives_in facts; values={[f.value for f in lives_in_currents]}"
    )
