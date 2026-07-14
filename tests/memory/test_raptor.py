import pytest

from genome.memory.facade import Memory
from genome.memory.graph import DERIVED_FROM
from genome.memory.raptor import (
    RAPTOR_OPERATOR,
    RaptorBuildResult,
    build_raptor_tree,
    search_raptor,
)
from tests.memory._fake_embed import FakeEmbeddingProvider


@pytest.fixture
def mem():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=32))
    yield m
    m.close()


def _seed_memories(mem, user="u"):
    topics = {
        "coffee": [
            "user loves pour-over coffee",
            "user drinks espresso every morning",
            "user roasts beans at home",
            "user prefers Ethiopian Yirgacheffe",
            "user owns a V60 dripper",
            "user avoids instant coffee",
        ],
        "travel": [
            "user just moved to Tokyo",
            "user visited Paris last year",
            "user loves traveling in Asia",
            "user plans a Kyoto trip next spring",
            "user hates long-haul flights",
            "user speaks conversational Japanese",
        ],
        "work": [
            "user works as a data scientist",
            "user leads a team of 5 engineers",
            "user uses Python for ML projects",
            "user writes research papers on LLMs",
            "user ships production features weekly",
            "user mentors junior engineers",
        ],
    }
    ids = {}
    for topic, facts in topics.items():
        ids[topic] = []
        for f in facts:
            rec = mem.add(f, user_id=user)[0]
            ids[topic].append(rec.id)
    return ids


def test_build_raptor_tree_creates_summaries(mem):
    _seed_memories(mem)
    result = build_raptor_tree(mem, user_id="u", branching_factor=3, max_levels=2)

    assert isinstance(result, RaptorBuildResult)
    assert result.levels >= 1
    assert result.summaries_created > 0

    # At least one summary memory exists
    summaries = [r for r in mem.list_all(user_id="u") if r.operator == RAPTOR_OPERATOR]
    assert len(summaries) > 0
    for s in summaries:
        assert s.metadata.get("raptor_level") == 1
        assert len(s.parents) >= 2
        assert s.operator == RAPTOR_OPERATOR


def test_build_raptor_with_explicit_summarizer(mem):
    _seed_memories(mem)
    captured_calls = []

    def my_summarizer(texts: list[str]) -> str:
        captured_calls.append(len(texts))
        return f"cluster-of-{len(texts)}"

    result = build_raptor_tree(
        mem, user_id="u", branching_factor=3, max_levels=1,
        summarizer=my_summarizer,
    )
    assert result.summaries_created > 0
    assert len(captured_calls) == result.summaries_created
    # All summaries use our format
    summaries = [r for r in mem.list_all(user_id="u") if r.operator == RAPTOR_OPERATOR]
    for s in summaries:
        assert s.content.startswith("cluster-of-")


def test_raptor_links_derivation_edges(mem):
    _seed_memories(mem)
    build_raptor_tree(mem, user_id="u", branching_factor=3, max_levels=1,
                     link_derivations=True)
    summaries = [r for r in mem.list_all(user_id="u") if r.operator == RAPTOR_OPERATOR]
    assert len(summaries) > 0
    # Each summary should have DERIVED_FROM edges to its members
    for s in summaries:
        derived = mem.related(s.id, relation=DERIVED_FROM, direction="out")
        assert len(derived) >= 2
        # Derived IDs match parents
        derived_ids = {d.id for d in derived}
        assert derived_ids == set(s.parents)


def test_raptor_respects_scope(mem):
    _seed_memories(mem, user="alice")
    _seed_memories(mem, user="bob")
    result = build_raptor_tree(mem, user_id="alice")
    # Only alice's summaries created
    alice_sums = [
        r for r in mem.list_all(user_id="alice") if r.operator == RAPTOR_OPERATOR
    ]
    bob_sums = [
        r for r in mem.list_all(user_id="bob") if r.operator == RAPTOR_OPERATOR
    ]
    assert len(alice_sums) == result.summaries_created
    assert len(bob_sums) == 0


def test_raptor_skips_when_too_few_memories(mem):
    mem.add("only fact", user_id="u")
    result = build_raptor_tree(mem, user_id="u", branching_factor=4)
    assert result.summaries_created == 0
    assert result.levels == 0


def test_search_raptor_at_level_zero_atomic(mem):
    _seed_memories(mem)
    build_raptor_tree(mem, user_id="u", branching_factor=3, max_levels=1)
    # At level 0, only atomic memories
    results = search_raptor(mem, "coffee", user_id="u", level=0, limit=5)
    for r in results:
        assert r.record.operator != RAPTOR_OPERATOR


def test_search_raptor_at_level_one_summaries(mem):
    _seed_memories(mem)
    build_raptor_tree(mem, user_id="u", branching_factor=3, max_levels=1)
    results = search_raptor(mem, "coffee", user_id="u", level=1, limit=5)
    # All returned are summaries at level 1
    for r in results:
        assert r.record.operator == RAPTOR_OPERATOR
        assert r.record.metadata.get("raptor_level") == 1


def test_memory_facade_shortcut(mem):
    _seed_memories(mem)
    # Use facade shortcut instead of standalone function
    result = mem.build_raptor_tree(user_id="u", branching_factor=3)
    assert result.summaries_created > 0
    # search_at_level also works via facade
    results = mem.search_at_level("coffee", user_id="u", level=0, limit=3)
    for r in results:
        assert r.record.operator != RAPTOR_OPERATOR


def test_raptor_with_llm_call(mem):
    _seed_memories(mem)

    def fake_llm(prompt: str) -> str:
        # Return the first line of prompt as-is (just for testing the path)
        return "LLM summary\n"

    result = build_raptor_tree(
        mem, user_id="u", branching_factor=3, max_levels=1, llm_call=fake_llm
    )
    assert result.summaries_created > 0
    summaries = [r for r in mem.list_all(user_id="u") if r.operator == RAPTOR_OPERATOR]
    for s in summaries:
        assert s.content == "LLM summary"


def test_raptor_rejects_invalid_branching_factor(mem):
    """branching_factor=0 would div-by-zero; bf=1 is meaningless. Refuse both."""
    _seed_memories(mem)
    for bad in (0, 1, -1):
        with pytest.raises(ValueError, match="branching_factor"):
            build_raptor_tree(mem, user_id="u", branching_factor=bad, max_levels=2)


def test_raptor_rejects_invalid_max_levels(mem):
    _seed_memories(mem)
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_levels"):
            build_raptor_tree(mem, user_id="u", branching_factor=3, max_levels=bad)
