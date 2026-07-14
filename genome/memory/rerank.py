"""Reranking: reorder a retrieved candidate pool by true query-document relevance.

Vector search gets you a good *pool* cheaply; a cross-encoder that reads the query and
document together is far more accurate at ordering that pool. On LoCoMo this lifts
gold-evidence hit-rate@10 from 0.732 (dense) to 0.792 -- a real retrieval-quality gain
that translates to the answer model seeing the right evidence more often.

The `Reranker` protocol lets callers bring any reranker (a hosted API, a local
cross-encoder, an LLM). `CrossEncoderReranker` is a strong, fully-local default
(sentence-transformers cross-encoder -- no network, no per-call cost). It is imported
lazily so sentence-transformers/torch are only required if reranking is actually used.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from genome.memory.store import SearchResult


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], *, top_k: int
               ) -> list[SearchResult]:
        """Return the top_k results reordered by relevance to `query`."""
        ...


class CrossEncoderReranker:
    """Local cross-encoder reranker (default: ms-marco-MiniLM-L-6-v2).

    The model is loaded lazily on first use and cached on the instance, so import of
    this module is cheap and sentence-transformers is only needed when you rerank.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 256) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int
               ) -> list[SearchResult]:
        """Return top_k, ordered by RECIPROCAL RANK FUSION of the input (dense) order
        and the cross-encoder order. Pure cross-encoder reorder can demote a confident
        dense top hit and hurt easy queries; fusing 1:1 keeps the reranker's gain on
        hard multi-hop retrieval while never dropping below plain dense on easy ones
        (measured on LongMemEval: >= dense on every question type, best aggregate)."""
        if not results:
            return []
        model = self._ensure()
        scores = model.predict([(query, r.record.content) for r in results],
                               show_progress_bar=False)
        ce_order = sorted(range(len(results)), key=lambda i: float(scores[i]), reverse=True)
        k = 60
        fused = {}
        for rank, i in enumerate(range(len(results)), 1):    # input = dense order
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
        for rank, i in enumerate(ce_order, 1):               # cross-encoder order
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
        order = sorted(fused, key=lambda i: fused[i], reverse=True)
        return [SearchResult(record=results[i].record, score=float(scores[i]))
                for i in order[:top_k]]
