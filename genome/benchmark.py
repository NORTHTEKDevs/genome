"""Professional benchmark harness for GENOME recombination validation.

Three benchmarks:

1. **Aggregate retrieval quality** (`benchmark_operators`): runs every operator over
   every parent pair, reports hit@k and precision@k with bootstrap 95% confidence
   intervals. Paired across operators (same pair, different operator) so statistical
   comparison is fair.

2. **Diversity (Criterion 2)** (`benchmark_diversity`): for each stochastic operator,
   runs the same parent pair with N different seeds. Measures fraction of pairs
   where the operator produces >= 50% non-overlapping top-5 retrievals across seeds
   -- the design-doc definition of "meaningfully different hybrids".

3. **Cross-encoder generalization** (`benchmark_encoders`): runs the aggregate
   benchmark against multiple sentence-transformer models. Answers the question
   "are these results encoder-specific or do they generalize?"

All benchmarks save structured JSON to `results/` and print a human-readable table.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from genome.corpus import RetrievalCorpus, build_default_corpus
from genome.dataset import ParentPair, load_parent_pairs
from genome.embeddings import EmbeddingProvider
from genome.metrics import any_hit_at_k, precision_at_k
from genome.operators import (
    OPERATORS,
    multi_point_crossover,
    single_point_crossover,
    uniform_crossover,
    uniform_crossover_with_mutation,
)

# Operators whose output depends on a random seed.
# uniform_crossover_with_mutation uses sigma=0.05 (the registry default) -- this
# is the diversity-tuned value that passes Criterion 2 while preserving retrieval.
STOCHASTIC_OPERATORS = {
    "single_point_crossover": lambda a, b, seed: single_point_crossover(a, b, seed=seed),
    "uniform_crossover": lambda a, b, seed: uniform_crossover(a, b, seed=seed),
    "uniform_crossover_with_mutation": lambda a, b, seed: uniform_crossover_with_mutation(
        a, b, seed=seed, sigma=0.05
    ),
    "multi_point_crossover": lambda a, b, seed: multi_point_crossover(
        a, b, seed=seed, num_points=3
    ),
}


@dataclass
class MetricWithCI:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def _bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> MetricWithCI:
    """95% percentile bootstrap CI of the mean."""
    if not values:
        return MetricWithCI(0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    return MetricWithCI(
        mean=float(arr.mean()),
        ci_low=float(np.percentile(means, 2.5)),
        ci_high=float(np.percentile(means, 97.5)),
        n=n,
    )


def benchmark_operators(
    provider: EmbeddingProvider | None = None,
    corpus: RetrievalCorpus | None = None,
    pairs: list[ParentPair] | None = None,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    filter_parents: bool = True,
    n_boot: int = 1000,
) -> dict[str, dict[str, MetricWithCI]]:
    """Aggregate retrieval benchmark with bootstrap CIs."""
    provider = provider or EmbeddingProvider()
    corpus = corpus or build_default_corpus(provider=provider)
    pairs = pairs or load_parent_pairs()

    parent_texts: list[str] = []
    for p in pairs:
        parent_texts.extend([p.parent_a, p.parent_b])
    parent_vecs = provider.encode_batch(parent_texts)

    per_pair: dict[str, dict[str, list[float]]] = {
        name: {f"hit@{k}": [] for k in ks} | {f"precision@{k}": [] for k in ks}
        for name in OPERATORS
    }

    for i, pair in enumerate(pairs):
        a = parent_vecs[2 * i]
        b = parent_vecs[2 * i + 1]
        pars = [pair.parent_a, pair.parent_b] if filter_parents else None
        for name, op in OPERATORS.items():
            hybrid = op(a, b)
            retrieval = corpus.search(hybrid, k=max(ks) + 5)
            for k in ks:
                per_pair[name][f"hit@{k}"].append(
                    any_hit_at_k(retrieval, pair.expected_hybrids, k, parents=pars)
                )
                per_pair[name][f"precision@{k}"].append(
                    precision_at_k(retrieval, pair.expected_hybrids, k, parents=pars)
                )

    out: dict[str, dict[str, MetricWithCI]] = {}
    for name, metric_lists in per_pair.items():
        out[name] = {
            metric: _bootstrap_ci(vals, n_boot=n_boot) for metric, vals in metric_lists.items()
        }
    return out


def benchmark_diversity(
    provider: EmbeddingProvider | None = None,
    corpus: RetrievalCorpus | None = None,
    pairs: list[ParentPair] | None = None,
    n_seeds: int = 5,
    k: int = 5,
    filter_parents: bool = True,
    threshold: float = 0.50,
) -> dict[str, dict[str, float]]:
    """Measure whether stochastic operators produce diverse hybrids across seeds.

    For each pair, run the operator with `n_seeds` different seeds, compute
    the top-k retrievals for each, and measure pairwise Jaccard overlap.
    Report mean overlap and fraction-of-pairs where mean overlap < `threshold`
    (the design-doc definition of "meaningfully different").
    """
    provider = provider or EmbeddingProvider()
    corpus = corpus or build_default_corpus(provider=provider)
    pairs = pairs or load_parent_pairs()

    parent_texts: list[str] = []
    for p in pairs:
        parent_texts.extend([p.parent_a, p.parent_b])
    parent_vecs = provider.encode_batch(parent_texts)

    out: dict[str, dict[str, float]] = {}
    for name, base_op in STOCHASTIC_OPERATORS.items():
        per_pair_mean_overlap: list[float] = []
        meaningfully_different_flags: list[int] = []
        for i, pair in enumerate(pairs):
            a = parent_vecs[2 * i]
            b = parent_vecs[2 * i + 1]
            pars_lc = {pair.parent_a.lower(), pair.parent_b.lower()} if filter_parents else set()
            topk_sets: list[set[str]] = []
            for s in range(n_seeds):
                hybrid = base_op(a, b, s)
                retrieval = corpus.search(hybrid, k=k + 5)
                filt = [
                    r.text.lower() for r in retrieval if r.text.lower() not in pars_lc
                ][:k]
                topk_sets.append(set(filt))
            overlaps = []
            for ii in range(len(topk_sets)):
                for jj in range(ii + 1, len(topk_sets)):
                    u = topk_sets[ii] | topk_sets[jj]
                    if not u:
                        overlaps.append(0.0)
                        continue
                    jac = len(topk_sets[ii] & topk_sets[jj]) / len(u)
                    overlaps.append(jac)
            mean_overlap = statistics.mean(overlaps) if overlaps else 1.0
            per_pair_mean_overlap.append(mean_overlap)
            meaningfully_different_flags.append(1 if mean_overlap < threshold else 0)
        out[name] = {
            "mean_overlap": statistics.mean(per_pair_mean_overlap),
            "fraction_meaningfully_different": statistics.mean(meaningfully_different_flags),
            "n_pairs": len(pairs),
            "n_seeds": n_seeds,
        }
    return out


def benchmark_encoders(
    model_names: list[str],
    pairs: list[ParentPair] | None = None,
    filter_parents: bool = True,
) -> dict[str, dict[str, dict[str, MetricWithCI]]]:
    """Run the aggregate benchmark across multiple encoders. Returns encoder -> op -> metric -> CI."""
    pairs = pairs or load_parent_pairs()
    out: dict[str, dict[str, dict[str, MetricWithCI]]] = {}
    for m in model_names:
        print(f"  encoder: {m}")
        provider = EmbeddingProvider(model_name=m)
        corpus = build_default_corpus(provider=provider)
        out[m] = benchmark_operators(
            provider=provider, corpus=corpus, pairs=pairs, filter_parents=filter_parents
        )
    return out


def save_benchmark(result: dict, path: Path | str) -> None:
    """Serialize benchmark result to JSON (MetricWithCI -> dict)."""
    def _cvt(x):
        if isinstance(x, MetricWithCI):
            return asdict(x)
        if isinstance(x, dict):
            return {k: _cvt(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_cvt(v) for v in x]
        return x
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(_cvt(result), indent=2))


def print_benchmark_table(result: dict[str, dict[str, MetricWithCI]], metric: str = "hit@3") -> None:
    """Pretty-print operator x metric table with CIs."""
    print(f"\n{metric} (95% CI, N=pairs):")
    ranked = sorted(result.items(), key=lambda kv: kv[1][metric].mean, reverse=True)
    print(f"  {'operator':36s} {'mean':>7s}  [  low, high ]  N")
    for name, metrics in ranked:
        m = metrics[metric]
        print(f"  {name:36s} {m.mean:>7.3f}  [{m.ci_low:.3f}, {m.ci_high:.3f}]  {m.n}")
