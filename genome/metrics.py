"""Evaluation metrics for hybrid retrieval quality.

Two metric families:

1. **Text-match metrics** (`precision_at_k`, `any_hit_at_k`): case-insensitive exact
   string matching. Fast, deterministic, but brittle -- "ML engineer" != "ML PM".

2. **Semantic-match metrics** (`semantic_precision_at_k`, `semantic_hit_at_k`):
   cosine similarity of the retrieved text embedding against any expected-hybrid
   embedding, with a threshold. Requires an encoder. Handles synonyms.

Both families support **parent filtering**: when a hybrid of A and B is retrieved,
A and B themselves are (correctly) highly similar to the hybrid in cosine space.
Counting them as retrievals is noise -- we want the hybrid's children, not the hybrid's
parents. Filtering removes parent strings from top-k before scoring.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from genome.corpus import RetrievalResult


def _filter_parents(
    results: list[RetrievalResult], parents: list[str] | None
) -> list[RetrievalResult]:
    if not parents:
        return results
    par_lc = {p.strip().lower() for p in parents}
    return [r for r in results if r.text.strip().lower() not in par_lc]


def precision_at_k(
    results: list[RetrievalResult],
    expected: list[str],
    k: int,
    parents: list[str] | None = None,
) -> float:
    """Precision over the RETURNED set: relevant / number-returned.

    NOTE ON DENOMINATOR: this divides by the number of results actually
    returned after parent-filtering and top-k truncation, NOT by k. So if
    filtering leaves 1 candidate and it is relevant, this returns 1.0 (not
    1/k). This is precision-over-retrieved, deliberately distinct from
    standard IR precision@k (which divides by k and penalizes under-
    retrieval). The headline retrieval numbers use `any_hit_at_k`; this metric
    is a secondary diagnostic. Keep the denominator in mind when quoting it.
    """
    expected_lc = {e.strip().lower() for e in expected}
    top_k = _filter_parents(results, parents)[:k]
    if not top_k:
        return 0.0
    matches = sum(1 for r in top_k if r.text.strip().lower() in expected_lc)
    return matches / len(top_k)


def any_hit_at_k(
    results: list[RetrievalResult],
    expected: list[str],
    k: int,
    parents: list[str] | None = None,
) -> float:
    """1.0 if ANY of (parent-filtered) top-k matches expected, else 0.0."""
    expected_lc = {e.strip().lower() for e in expected}
    for r in _filter_parents(results, parents)[:k]:
        if r.text.strip().lower() in expected_lc:
            return 1.0
    return 0.0


def semantic_hit_at_k(
    results: list[RetrievalResult],
    expected_vecs: np.ndarray,
    k: int,
    threshold: float = 0.70,
    parents: list[str] | None = None,
    corpus_vecs: np.ndarray | None = None,
) -> float:
    """1.0 if any of top-k (parent-filtered) retrieved vectors has cosine >= threshold
    against any expected-hybrid embedding. Requires the corpus embeddings.
    """
    if corpus_vecs is None:
        raise ValueError("corpus_vecs required for semantic matching")
    top_k = _filter_parents(results, parents)[:k]
    if not top_k:
        return 0.0
    exp_norm = expected_vecs / np.maximum(
        np.linalg.norm(expected_vecs, axis=1, keepdims=True), 1e-12
    )
    for r in top_k:
        v = corpus_vecs[r.index]
        v_norm = v / max(np.linalg.norm(v), 1e-12)
        sims = exp_norm @ v_norm
        if float(sims.max()) >= threshold:
            return 1.0
    return 0.0


def evaluate_hybrid(
    results: list[RetrievalResult],
    expected: list[str],
    ks: tuple[int, ...] = (1, 3, 5, 10),
    parents: list[str] | None = None,
) -> dict[str, float]:
    """Compute precision@k and any-hit@k for multiple k values, optionally filtering parents."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(results, expected, k, parents=parents)
        out[f"hit@{k}"] = any_hit_at_k(results, expected, k, parents=parents)
    return out
