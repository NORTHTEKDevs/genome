import pytest

from genome.errors import (
    GenomeError,
    MemoryNotFoundError,
    OperatorError,
    SynthesisError,
)
from genome.memory.facade import Memory
from tests.memory._fake_embed import FakeEmbeddingProvider


def test_genome_error_inherits_from_valueerror():
    """Back-compat: old code catching ValueError still works."""
    e = GenomeError("msg")
    assert isinstance(e, ValueError)


def test_memory_not_found_carries_hint():
    try:
        raise MemoryNotFoundError("mem_xyz")
    except MemoryNotFoundError as e:
        assert "mem_xyz" in str(e)
        assert "Hint:" in str(e)


def test_synthesis_error_has_hint():
    e = SynthesisError("too few parents", hint="pass 2+ ids")
    assert "too few parents" in str(e)
    assert "Hint: pass 2+ ids" in str(e)


def test_operator_error_is_genome_error():
    assert issubclass(OperatorError, GenomeError)


# ---------- Memory raises structured errors ----------

@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    yield m
    m.close()


def test_synthesize_missing_parent_raises_memory_not_found(mem):
    a = mem.add("a", user_id="u")[0]
    with pytest.raises(MemoryNotFoundError) as ei:
        mem.synthesize(memory_ids=[a.id, "mem_missing"], user_id="u")
    assert "mem_missing" in str(ei.value)


def test_synthesize_too_few_parents_raises_synthesis_error(mem):
    a = mem.add("a", user_id="u")[0]
    with pytest.raises(SynthesisError) as ei:
        mem.synthesize(memory_ids=[a.id], user_id="u")
    assert "2 parent" in str(ei.value).lower()


def test_link_missing_endpoint_raises_memory_not_found(mem):
    a = mem.add("a", user_id="u")[0]
    with pytest.raises(MemoryNotFoundError):
        mem.link(a.id, "mem_nope", relation="rel")


def test_extract_entities_missing_raises(mem):
    with pytest.raises(MemoryNotFoundError):
        mem.extract_entities("mem_nope")


def test_structured_errors_still_catchable_as_valueerror(mem):
    """Confirms backward compatibility."""
    a = mem.add("a", user_id="u")[0]
    with pytest.raises(ValueError):
        mem.synthesize(memory_ids=[a.id], user_id="u")
