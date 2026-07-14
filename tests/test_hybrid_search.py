"""Tests for hybrid BM25 + dense retrieval via Reciprocal Rank Fusion."""

from genome.memory.hybrid import HybridScorer, reciprocal_rank_fusion


def test_rrf_basic():
    """RRF combines two ranked lists with k=60 default.

    By Jensen's inequality on f(x)=1/(k+x), items at rank-extremes (1 and 3)
    score higher than items at the middle (2,2). So a (1,3) and c (3,1) both
    outrank b (2,2).
    """
    dense = ["a", "b", "c"]
    sparse = ["c", "b", "a"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert set(fused) == {"a", "b", "c"}
    assert fused[-1] == "b"
    assert fused.index("a") < fused.index("b")
    assert fused.index("c") < fused.index("b")


def test_rrf_handles_disjoint_lists():
    dense = ["a", "b"]
    sparse = ["c", "d"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    assert set(fused) == {"a", "b", "c", "d"}


def test_hybrid_scorer_falls_back_to_dense_when_no_corpus():
    scorer = HybridScorer()
    dense_results = [("id1", 0.9), ("id2", 0.8)]
    fused = scorer.fuse(query="anything", dense_results=dense_results, corpus={})
    assert [r[0] for r in fused] == ["id1", "id2"]


def test_hybrid_scorer_boosts_keyword_match():
    """BM25 keyword match should pull a record up over a non-matching dense-#1.

    Dense ranking has id2 first (semantic confusion). BM25 ranks id1/id3 high
    because they contain "Tokyo". After RRF, id1 should outrank id2 because
    its average rank across both rankings is better.
    """
    corpus = {
        "id1": "the user lives in Tokyo",
        "id2": "the user enjoys hiking",
        "id3": "Tokyo is a city in Japan",
    }
    dense_results = [("id2", 0.85), ("id1", 0.80), ("id3", 0.75)]
    scorer = HybridScorer()
    fused = scorer.fuse(query="Tokyo", dense_results=dense_results, corpus=corpus)
    fused_ids = [r[0] for r in fused]
    assert fused_ids[0] == "id1", f"BM25 should boost id1 to top, got {fused_ids}"
    assert fused_ids.index("id1") < fused_ids.index("id2"), (
        f"keyword match id1 should outrank dense-#1 id2, got {fused_ids}"
    )


def test_memory_search_hybrid_mode_works():
    """Memory.search(mode='hybrid') uses BM25+dense fusion."""
    from genome import Memory
    m = Memory()
    m.add("user lives in Tokyo", user_id="alice")
    m.add("user enjoys hiking on weekends", user_id="alice")
    m.add("Tokyo has great food", user_id="alice")
    results_hybrid = m.search("Tokyo", user_id="alice", limit=3, mode="hybrid")
    hybrid_contents = [r.content for r in results_hybrid[:2]]
    assert any("Tokyo" in c for c in hybrid_contents), "hybrid lost the keyword match"
    assert len(results_hybrid) <= 3


def test_memory_search_dense_mode_is_default():
    """Memory.search() with no mode argument uses dense (preserves v2.0 behavior)."""
    from genome import Memory
    m = Memory()
    m.add("user lives in Tokyo", user_id="alice")
    results = m.search("Tokyo", user_id="alice", limit=1)  # no mode arg
    assert len(results) == 1


def test_memory_search_invalid_mode_raises():
    import pytest

    from genome import Memory

    m = Memory()
    with pytest.raises(ValueError, match="mode must be"):
        m.search("anything", user_id="alice", mode="elastic")
