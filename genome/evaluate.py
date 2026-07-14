"""End-to-end evaluation harness.

For each recombination operator:
  for each parent pair:
    - embed parent_a, parent_b
    - apply operator to produce hybrid
    - retrieve top-k from corpus
    - compute precision@k, hit@k vs expected_hybrids (parents filtered by default)
  aggregate mean across pairs

Returns a dict mapping operator_name -> aggregated metrics.

Parent filtering is ON by default. When a hybrid of "coffee" + "milk" retrieves
"coffee" and "milk" as top-2, those are correct in cosine terms but uninformative
for hybrid evaluation (they're the parents, not hybrids). Filtering removes them
before scoring. Pass `filter_parents=False` to disable.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import statistics

from genome.corpus import RetrievalCorpus, build_default_corpus
from genome.dataset import ParentPair, load_parent_pairs
from genome.embeddings import EmbeddingProvider
from genome.metrics import evaluate_hybrid
from genome.operators import OPERATORS


def run_evaluation(
    pairs: list[ParentPair] | None = None,
    corpus: RetrievalCorpus | None = None,
    provider: EmbeddingProvider | None = None,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    limit_pairs: int | None = None,
    filter_parents: bool = True,
    operators: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Run all operators against all parent pairs and aggregate metrics.

    Args:
        pairs: Parent pairs to evaluate. Defaults to the full curated set.
        corpus: Retrieval corpus. Defaults to the built-in 370-item corpus.
        provider: Embedding provider. Defaults to all-MiniLM-L6-v2.
        ks: Top-k values to report precision@k and hit@k for.
        limit_pairs: If set, evaluate only the first N pairs (for smoke tests).
        filter_parents: If True (default), filter parent strings out of retrievals
            before scoring. Prevents parents from crowding out hybrids in top-k.
        operators: Optional override of the OPERATORS registry (name -> callable).
    """
    provider = provider or EmbeddingProvider()
    corpus = corpus or build_default_corpus(provider=provider)
    pairs = pairs or load_parent_pairs()
    if limit_pairs is not None:
        pairs = pairs[:limit_pairs]

    op_map = operators if operators is not None else OPERATORS

    parent_texts: list[str] = []
    for p in pairs:
        parent_texts.extend([p.parent_a, p.parent_b])
    parent_vecs = provider.encode_batch(parent_texts)

    results: dict[str, dict[str, list[float]]] = {
        op_name: {f"precision@{k}": [] for k in ks} | {f"hit@{k}": [] for k in ks}
        for op_name in op_map
    }

    for i, pair in enumerate(pairs):
        a_vec = parent_vecs[2 * i]
        b_vec = parent_vecs[2 * i + 1]
        pars = [pair.parent_a, pair.parent_b] if filter_parents else None
        for op_name, op in op_map.items():
            hybrid = op(a_vec, b_vec)
            # Retrieve extra so that after parent filtering we still have max(ks)
            retrieval = corpus.search(hybrid, k=max(ks) + 5)
            metrics = evaluate_hybrid(retrieval, pair.expected_hybrids, ks=ks, parents=pars)
            for k, v in metrics.items():
                results[op_name][k].append(v)

    aggregated: dict[str, dict[str, float]] = {}
    for op_name, metric_lists in results.items():
        aggregated[op_name] = {
            k: statistics.mean(vals) if vals else 0.0
            for k, vals in metric_lists.items()
        }
    return aggregated
