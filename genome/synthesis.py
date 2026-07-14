"""N-parent recombination for the memory layer.

The v0.1-v0.2 operators take two parents. For memory synthesis we often want to
merge N memories at once. This module generalizes each operator to N parents.

Uniform and frequency operators extend naturally (pick dim from best of N).
Crossover operators fold pairwise through the operator. Averaging is the mean
over N. Attention-weighted extends to N-way softmax.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from genome.operators import (
    single_point_crossover,
)


def _stack(parents: list[np.ndarray]) -> np.ndarray:
    if not parents:
        raise ValueError("need at least one parent")
    arr = np.stack([p.astype(np.float32) for p in parents])
    if arr.ndim != 2:
        raise ValueError("parents must be 1-D vectors of equal shape")
    return arr


def average_n(parents: list[np.ndarray]) -> np.ndarray:
    """Element-wise mean of N parents."""
    return _stack(parents).mean(axis=0).astype(np.float32)


def uniform_crossover_n(
    parents: list[np.ndarray], seed: int | None = None
) -> np.ndarray:
    """For each dim, pick the value from one of the N parents uniformly at random."""
    stacked = _stack(parents)
    n_parents, dim = stacked.shape
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, n_parents, size=dim)
    out = stacked[choices, np.arange(dim)]
    return out.astype(np.float32)


def frequency_crossover_n(parents: list[np.ndarray]) -> np.ndarray:
    """For each dim, pick the value from whichever parent has the largest |value|.

    Ties go to the earlier parent (stable argmax via numpy's behavior).
    """
    stacked = _stack(parents)
    # argmax over parents axis for |stacked|
    idx = np.argmax(np.abs(stacked), axis=0)
    out = stacked[idx, np.arange(stacked.shape[1])]
    return out.astype(np.float32)


def attention_weighted_n(
    parents: list[np.ndarray], temperature: float = 1.0
) -> np.ndarray:
    """Per-dim N-way softmax over |values| yields blend weights across parents."""
    stacked = _stack(parents)
    scaled = np.abs(stacked) / max(temperature, 1e-12)
    scaled = scaled - scaled.max(axis=0, keepdims=True)
    weights = np.exp(scaled)
    weights = weights / weights.sum(axis=0, keepdims=True)
    return (weights * stacked).sum(axis=0).astype(np.float32)


def gaussian_mutation_n(
    vec: np.ndarray, sigma: float = 0.05, seed: int | None = None
) -> np.ndarray:
    """Apply Gaussian noise to an already-synthesized hybrid. Same as 2-parent version
    but exposed here for API symmetry with the N-parent family."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(vec.shape).astype(np.float32) * sigma
    return (vec + noise).astype(np.float32)


def uniform_crossover_with_mutation_n(
    parents: list[np.ndarray], sigma: float = 0.05, seed: int | None = None
) -> np.ndarray:
    """N-parent uniform crossover + Gaussian mutation."""
    hybrid = uniform_crossover_n(parents, seed=seed)
    # Distinct sub-seed for the noise draw so mask and noise are independent
    # (still reproducible for a given seed).
    mut_seed = (
        None if seed is None
        else int(np.random.default_rng(seed).integers(0, 2**31 - 1))
    )
    return gaussian_mutation_n(hybrid, sigma=sigma, seed=mut_seed)


def _pairwise_fold(
    parents: list[np.ndarray],
    op,
    **kw,
) -> np.ndarray:
    """Fold parents left-to-right through a 2-parent operator."""
    result = parents[0].astype(np.float32)
    for p in parents[1:]:
        result = op(result, p, **kw)
    return result.astype(np.float32)


def single_point_crossover_n(
    parents: list[np.ndarray], seed: int | None = None
) -> np.ndarray:
    """Fold N parents through single-point crossover (each step picks a fresh split)."""
    if not parents:
        # Match the standard ValueError from _stack instead of a raw IndexError
        # on parents[0], so recombine([], ...) fails consistently.
        raise ValueError("need at least one parent")
    _stack(parents)  # validate equal shapes / 1-D, consistent with other ops
    if len(parents) == 1:
        return parents[0].astype(np.float32)
    rng = np.random.default_rng(seed)
    result = parents[0].astype(np.float32)
    for p in parents[1:]:
        sub_seed = int(rng.integers(0, 2**31 - 1))
        result = single_point_crossover(result, p, seed=sub_seed)
    return result.astype(np.float32)


# Every entry accepts **kw and forwards only the args it supports, so a caller
# (e.g. Memory.synthesize) can pass a uniform `seed`/`temperature` across all
# operators without a TypeError from the ones that ignore it.
N_PARENT_OPERATORS: dict[str, Callable[..., np.ndarray]] = {
    "simple_average": lambda parents, **kw: average_n(parents),
    "uniform_crossover": lambda parents, **kw: uniform_crossover_n(
        parents, seed=kw.get("seed")
    ),
    "frequency_crossover": lambda parents, **kw: frequency_crossover_n(parents),
    "attention_weighted_crossover": lambda parents, **kw: attention_weighted_n(
        parents, temperature=kw.get("temperature", 1.0)
    ),
    "uniform_crossover_with_mutation": lambda parents, **kw: uniform_crossover_with_mutation_n(
        parents, sigma=kw.get("sigma", 0.05), seed=kw.get("seed")
    ),
    "single_point_crossover": lambda parents, **kw: single_point_crossover_n(
        parents, seed=kw.get("seed")
    ),
}
"""Registry of N-parent operators. Keys match the 2-parent registry in genome.operators."""


def recombine(
    parents: list[np.ndarray],
    operator: str = "uniform_crossover",
    **kw,
) -> np.ndarray:
    """Recombine N parent embeddings via the named operator.

    Parameters
    ----------
    parents : list of 1-D numpy arrays (all same shape, dtype float32 recommended)
    operator : one of the keys in N_PARENT_OPERATORS
    **kw : operator-specific kwargs (temperature, sigma, seed, ...)
    """
    if operator not in N_PARENT_OPERATORS:
        raise ValueError(
            f"Unknown operator {operator!r}. "
            f"Available: {sorted(N_PARENT_OPERATORS)}"
        )
    return N_PARENT_OPERATORS[operator](parents, **kw)


__all__ = [
    "recombine",
    "N_PARENT_OPERATORS",
    "average_n",
    "uniform_crossover_n",
    "frequency_crossover_n",
    "attention_weighted_n",
    "gaussian_mutation_n",
    "uniform_crossover_with_mutation_n",
    "single_point_crossover_n",
]
