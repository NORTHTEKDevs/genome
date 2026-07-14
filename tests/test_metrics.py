import numpy as np

from genome.corpus import RetrievalResult
from genome.metrics import (
    any_hit_at_k,
    precision_at_k,
    semantic_hit_at_k,
)


def test_precision_at_k_perfect_match():
    results = [
        RetrievalResult("technical product manager", 0.9, 0),
        RetrievalResult("AI product manager", 0.85, 1),
        RetrievalResult("distractor", 0.5, 2),
    ]
    expected = ["technical product manager", "AI product manager"]
    assert precision_at_k(results, expected, k=2) == 1.0


def test_precision_at_k_partial_match():
    results = [
        RetrievalResult("technical product manager", 0.9, 0),
        RetrievalResult("distractor_a", 0.5, 1),
        RetrievalResult("distractor_b", 0.4, 2),
        RetrievalResult("distractor_c", 0.3, 3),
    ]
    expected = ["technical product manager", "AI product manager"]
    assert precision_at_k(results, expected, k=4) == 0.25


def test_precision_at_k_no_match():
    results = [
        RetrievalResult("nope", 0.9, 0),
        RetrievalResult("nothing", 0.85, 1),
    ]
    expected = ["technical product manager"]
    assert precision_at_k(results, expected, k=2) == 0.0


def test_precision_at_k_case_insensitive_exact():
    results = [RetrievalResult("Technical Product Manager", 0.9, 0)]
    expected = ["technical product manager"]
    assert precision_at_k(results, expected, k=1) == 1.0


def test_precision_at_k_filters_parents():
    results = [
        RetrievalResult("coffee", 0.95, 0),       # parent - filtered
        RetrievalResult("milk", 0.94, 1),         # parent - filtered
        RetrievalResult("latte", 0.88, 2),        # hybrid
        RetrievalResult("distractor", 0.5, 3),
    ]
    expected = ["latte", "cappuccino"]
    parents = ["coffee", "milk"]
    # Without filtering: 1/2 in top-2 (coffee is the first match, which isn't expected, so 0/2)
    # With filtering: top-2 is latte + distractor, 1/2 = 0.5
    assert precision_at_k(results, expected, k=2, parents=parents) == 0.5


def test_any_hit_at_k_filters_parents():
    results = [
        RetrievalResult("coffee", 0.95, 0),
        RetrievalResult("milk", 0.94, 1),
        RetrievalResult("latte", 0.88, 2),
    ]
    expected = ["latte"]
    parents = ["coffee", "milk"]
    # Without filtering, top-2 = [coffee, milk], no hit
    # With filtering, top-2 = [latte, ...], hit
    assert any_hit_at_k(results, expected, k=2, parents=parents) == 1.0


def test_semantic_hit_at_k_threshold():
    # corpus has 3 items, indices 0-2
    corpus_vecs = np.array(
        [
            [1.0, 0.0, 0.0],   # distractor
            [0.9, 0.1, 0.0],   # close to expected
            [0.0, 1.0, 0.0],   # exact expected
        ],
        dtype=np.float32,
    )
    expected_vecs = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    # top-2 contains index 2 (exact match) -> should hit at threshold 0.9
    results = [
        RetrievalResult("distractor", 0.9, 0),
        RetrievalResult("exact", 0.8, 2),
    ]
    assert semantic_hit_at_k(
        results, expected_vecs, k=2, threshold=0.9, corpus_vecs=corpus_vecs
    ) == 1.0
    # If threshold is 1.01 (impossible), no hit
    assert semantic_hit_at_k(
        results, expected_vecs, k=2, threshold=1.01, corpus_vecs=corpus_vecs
    ) == 0.0
