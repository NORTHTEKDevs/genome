import time

import numpy as np

from genome.memory.consolidation import ConsolidationResult, score_memories
from genome.memory.facade import Memory
from genome.memory.schema import MemoryRecord
from tests.memory._fake_embed import FakeEmbeddingProvider


def _make_rec(content: str, **kw) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        embedding=np.random.default_rng(abs(hash(content)) % 1000).standard_normal(8).astype(np.float32),
        **kw,
    )


def test_score_memories_sorts_by_fitness():
    now = time.time()
    new_heavy = _make_rec("recent and used", created_at=now - 86400, access_count=50)
    old_unused = _make_rec("old unused", created_at=now - 86400 * 100, access_count=0)
    scored = score_memories([new_heavy, old_unused], half_life_days=30.0, now=now)
    by_rec = {r.id: s for r, s in scored}
    assert by_rec[new_heavy.id] > by_rec[old_unused.id]


def test_consolidate_noop_below_max():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        for i in range(5):
            m.add(f"fact {i}", user_id="u")
        result = m.consolidate(user_id="u", max_memories=10)
        assert isinstance(result, ConsolidationResult)
        assert result.before == 5
        assert result.pruned == 0
        assert result.kept == 5
        assert result.synthesized == 0
    finally:
        m.close()


def test_consolidate_prunes_to_max():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        for i in range(10):
            m.add(f"fact {i}", user_id="u")
        result = m.consolidate(user_id="u", max_memories=5)
        assert result.before == 10
        assert result.kept == 5
        assert result.pruned == 5
        assert m.count(user_id="u") == 5
    finally:
        m.close()


def test_consolidate_synthesizes_before_pruning():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        for i in range(10):
            m.add(f"fact {i}", user_id="u")
        result = m.consolidate(
            user_id="u",
            max_memories=5,
            synthesize_before_prune=True,
        )
        # 5 pruned, so 2 hybrid pairs created (floor(5/2) = 2)
        assert result.synthesized == 2
        # Final count = 5 kept + 2 hybrids = 7
        assert m.count(user_id="u") == 7
        # Hybrids are flagged in metadata
        synthesized = [r for r in m.list_all(user_id="u") if r.metadata.get("consolidation")]
        assert len(synthesized) == 2
    finally:
        m.close()


def test_consolidate_respects_scope():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        for i in range(6):
            m.add(f"a{i}", user_id="alice")
        for i in range(6):
            m.add(f"b{i}", user_id="bob")
        result = m.consolidate(user_id="alice", max_memories=3)
        assert result.before == 6
        assert result.kept == 3
        # Bob's memories untouched
        assert m.count(user_id="bob") == 6
    finally:
        m.close()


def test_consolidate_keeps_high_access_count():
    m = Memory(embedding_provider=FakeEmbeddingProvider(dim=8))
    try:
        important = m.add("important fact", user_id="u")[0]
        # Access this memory repeatedly
        for _ in range(20):
            m.get(important.id)
        # Add more memories with no access
        for i in range(9):
            m.add(f"filler {i}", user_id="u")
        result = m.consolidate(user_id="u", max_memories=3)
        assert result.kept == 3
        # The heavily-accessed memory should survive
        surviving = m.list_all(user_id="u")
        surviving_ids = {r.id for r in surviving}
        assert important.id in surviving_ids
    finally:
        m.close()
