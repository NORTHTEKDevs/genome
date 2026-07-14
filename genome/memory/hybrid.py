"""Hybrid BM25 + dense retrieval via Reciprocal Rank Fusion.

Pure cosine search misses on entity names, dates, numbers, and rare terms.
BM25 alone misses on paraphrases. RRF combines both rankings without
needing per-query score normalization.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

from collections.abc import Iterable

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace + lowercase tokenizer. Adequate for BM25."""
    return text.lower().split()


def reciprocal_rank_fusion(
    rankings: Iterable[list[str]], k: int = 60
) -> list[str]:
    """RRF with default k=60 (per Cormack et al. 2009).

    Score for item d = sum over rankings r of 1 / (k + rank_r(d))
    where rank_r(d) is 1-based rank in ranking r.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


class HybridScorer:
    """Builds a transient BM25 index per fuse() call. Stateless.

    Callers pass a `corpus: dict[id, text]` so we never need to know about
    the underlying MemoryStore. Index lifetime is one query.
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        *,
        query: str,
        dense_results: list[tuple[str, float]],
        corpus: dict[str, str],
        sparse_topk: int = 50,
    ) -> list[tuple[str, float]]:
        """Fuse dense top-N with BM25 top-N over `corpus` via RRF.

        Returns list of (id, fused_score) sorted descending.
        """
        dense_ranking = [r[0] for r in dense_results]
        if not corpus:
            return dense_results

        ids = list(corpus.keys())
        # Defense in depth: rank-bm25 raises ZeroDivisionError when the
        # entire corpus is empty-token. Replace any empty token list with a
        # per-call UUID placeholder so the IDF math is stable AND so the
        # placeholder cannot collide with any real document content.
        # (A static "__empty__" string would silently corrupt rankings on
        # documents containing that literal token.)
        import uuid as _uuid
        placeholder = f"__bm25_placeholder_{_uuid.uuid4().hex[:8]}__"
        tokenized = [_tokenize(corpus[i]) or [placeholder] for i in ids]
        bm25 = BM25Okapi(tokenized)
        q_tokens = _tokenize(query) or [placeholder]
        bm25_scores = bm25.get_scores(q_tokens)
        ranked_pairs = sorted(
            zip(ids, bm25_scores, strict=True), key=lambda x: x[1], reverse=True
        )
        sparse_ranking = [pid for pid, _ in ranked_pairs[:sparse_topk]]

        fused_ids = reciprocal_rank_fusion(
            [dense_ranking, sparse_ranking], k=self.k
        )
        scores: dict[str, float] = {}
        for ranking in (dense_ranking, sparse_ranking):
            for rank, item in enumerate(ranking, start=1):
                scores[item] = scores.get(item, 0.0) + 1.0 / (self.k + rank)
        return [(fid, scores[fid]) for fid in fused_ids]
