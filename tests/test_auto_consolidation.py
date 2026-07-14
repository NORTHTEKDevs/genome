"""Tests for auto-consolidation trigger on memory count threshold."""

from genome import Memory
from genome.memory.extraction import IdentityExtractor


def test_auto_consolidate_triggers_at_threshold():
    """Memory(auto_consolidate_threshold=10) prunes when count exceeds 10."""
    m = Memory(
        extractor=IdentityExtractor(),
        auto_consolidate_threshold=10,
        auto_consolidate_target=5,
        auto_consolidate_synthesize=False,
    )
    for i in range(15):
        m.add(f"fact number {i}", user_id="alice")
    count = m.count(user_id="alice")
    assert count <= 10, f"expected <=10 after auto-consolidation, got {count}"


def test_auto_consolidate_creates_hybrids_when_synthesize_on():
    """Auto-consolidate with synthesize_before_prune=True creates hybrid records."""
    m = Memory(
        extractor=IdentityExtractor(),
        auto_consolidate_threshold=10,
        auto_consolidate_target=5,
        auto_consolidate_synthesize=True,
    )
    for i in range(15):
        m.add(f"fact number {i}", user_id="alice")
    all_recs = m.list_all(user_id="alice")
    hybrids = [r for r in all_recs if r.parents]
    assert len(hybrids) >= 1, "expected at least one synthesized hybrid"


def test_auto_consolidate_disabled_by_default():
    """Without threshold, auto-consolidation does not fire."""
    m = Memory(extractor=IdentityExtractor())
    for i in range(50):
        m.add(f"fact number {i}", user_id="alice")
    assert m.count(user_id="alice") == 50


def test_auto_consolidate_per_scope():
    """Different scopes consolidate independently of each other."""
    m = Memory(
        extractor=IdentityExtractor(),
        auto_consolidate_threshold=10,
        auto_consolidate_target=5,
        auto_consolidate_synthesize=False,
    )
    for i in range(15):
        m.add(f"alice fact {i}", user_id="alice")
    # bob has only 3 memories so should NOT trigger consolidation
    for i in range(3):
        m.add(f"bob fact {i}", user_id="bob")
    assert m.count(user_id="alice") <= 10
    assert m.count(user_id="bob") == 3
