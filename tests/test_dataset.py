from genome.dataset import ParentPair, load_parent_pairs


def test_load_parent_pairs_returns_list():
    pairs = load_parent_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) >= 20


def test_parent_pair_structure():
    pairs = load_parent_pairs()
    first = pairs[0]
    assert isinstance(first, ParentPair)
    assert isinstance(first.id, str)
    assert isinstance(first.parent_a, str)
    assert isinstance(first.parent_b, str)
    assert isinstance(first.expected_hybrids, list)
    assert len(first.expected_hybrids) >= 1
    assert all(isinstance(h, str) for h in first.expected_hybrids)
