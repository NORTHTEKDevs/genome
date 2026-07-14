"""Recombination operators for GENOME.

Each operator takes two parent embeddings A, B (1-D numpy arrays of
equal shape) and returns a hybrid embedding of the same shape.
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np


def _assert_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    """Reject mismatched parent shapes loudly.

    numpy would otherwise BROADCAST a length-1 (or otherwise compatible) parent
    against the other, silently producing a corrupt hybrid instead of an error
    -- e.g. recombining a 384-d and a 768-d embedding from two different
    encoders. Every operator here promises a same-shape output, so enforce it.
    """
    if a.shape != b.shape:
        raise ValueError(
            f"parent embeddings must have the same shape, got {a.shape} vs "
            f"{b.shape} (are you mixing embedding models / dimensions?)"
        )


def simple_average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Baseline: element-wise mean of the two parents."""
    _assert_same_shape(a, b)
    return ((a + b) / 2.0).astype(np.float32)


def weighted_sum(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Baseline: alpha * a + (1 - alpha) * b."""
    _assert_same_shape(a, b)
    return (alpha * a + (1.0 - alpha) * b).astype(np.float32)


def concat_project(
    a: np.ndarray, b: np.ndarray, projection_seed: int = 0
) -> np.ndarray:
    """Baseline: concatenate parents then project back to parent dim via random matrix.

    A fixed random projection makes this deterministic per seed.
    """
    _assert_same_shape(a, b)
    rng = np.random.default_rng(projection_seed)
    concat = np.concatenate([a, b])
    projection = rng.standard_normal((concat.shape[0], a.shape[0])).astype(np.float32)
    norms = np.linalg.norm(projection, axis=0, keepdims=True)
    # Floor near-zero column norms to prevent overflow on degenerate seeds.
    norms = np.maximum(norms, 1e-12)
    projection /= norms
    return (concat @ projection).astype(np.float32)


def single_point_crossover(
    a: np.ndarray,
    b: np.ndarray,
    crossover_point: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Classical GA single-point crossover.

    Take a[:k] and b[k:] where k is the crossover point.
    If crossover_point is None, choose k uniformly at random using seed.
    """
    _assert_same_shape(a, b)
    if crossover_point is None:
        rng = np.random.default_rng(seed)
        crossover_point = int(rng.integers(0, a.shape[0] + 1))
    result = np.empty_like(a, dtype=np.float32)
    result[:crossover_point] = a[:crossover_point]
    result[crossover_point:] = b[crossover_point:]
    return result


def uniform_crossover(
    a: np.ndarray,
    b: np.ndarray,
    seed: int | None = None,
    prob_a: float = 0.5,
) -> np.ndarray:
    """Uniform crossover: each dim independently from A with prob_a, else B."""
    _assert_same_shape(a, b)
    rng = np.random.default_rng(seed)
    mask = rng.random(a.shape) < prob_a
    return np.where(mask, a, b).astype(np.float32)


def frequency_crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Frequency-based: keep the dim from whichever parent has larger |value|.

    Ties go to A.
    """
    _assert_same_shape(a, b)
    mask = np.abs(a) >= np.abs(b)
    return np.where(mask, a, b).astype(np.float32)


def attention_weighted_crossover(
    a: np.ndarray, b: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    """Per-dim softmax over |a|, |b| yields blend weights.

    High temperature -> uniform weights -> near-average.
    Low temperature -> sharp weights -> near-frequency-crossover.
    """
    _assert_same_shape(a, b)
    stacked = np.stack([np.abs(a), np.abs(b)]) / max(temperature, 1e-12)
    # Subtract max for numerical stability
    stacked = stacked - stacked.max(axis=0, keepdims=True)
    weights = np.exp(stacked)
    weights = weights / weights.sum(axis=0, keepdims=True)
    return (weights[0] * a + weights[1] * b).astype(np.float32)


def gaussian_mutation(
    vec: np.ndarray, sigma: float = 0.01, seed: int | None = None
) -> np.ndarray:
    """Add Gaussian noise with stdev sigma."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(vec.shape).astype(np.float32) * sigma
    return (vec + noise).astype(np.float32)


def uniform_crossover_with_mutation(
    a: np.ndarray,
    b: np.ndarray,
    seed: int | None = None,
    prob_a: float = 0.5,
    sigma: float = 0.01,
) -> np.ndarray:
    """Uniform crossover followed by Gaussian mutation."""
    hybrid = uniform_crossover(a, b, seed=seed, prob_a=prob_a)
    # Derive a distinct sub-seed for the noise draw so the crossover mask and
    # the mutation noise are statistically independent (still fully
    # reproducible for a given seed), rather than both restarting from `seed`.
    mut_seed = (
        None if seed is None
        else int(np.random.default_rng(seed).integers(0, 2**31 - 1))
    )
    return gaussian_mutation(hybrid, sigma=sigma, seed=mut_seed)


def multi_point_crossover(
    a: np.ndarray,
    b: np.ndarray,
    num_points: int = 2,
    seed: int | None = None,
) -> np.ndarray:
    """K-point crossover: pick num_points crossover points and alternate parents."""
    _assert_same_shape(a, b)
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    if num_points == 0:
        return a.astype(np.float32).copy()
    points = sorted(rng.choice(n, size=min(num_points, n), replace=False).tolist())
    result = np.empty_like(a, dtype=np.float32)
    use_a = True
    prev = 0
    for p in points:
        result[prev:p] = a[prev:p] if use_a else b[prev:p]
        use_a = not use_a
        prev = p
    result[prev:] = a[prev:] if use_a else b[prev:]
    return result


# Registry with sensible defaults for operators that take extra args
OPERATORS = {
    "simple_average": simple_average,
    "weighted_sum": lambda a, b: weighted_sum(a, b, alpha=0.5),
    "concat_project": lambda a, b: concat_project(a, b, projection_seed=42),
    "single_point_crossover": lambda a, b: single_point_crossover(a, b, seed=42),
    "uniform_crossover": lambda a, b: uniform_crossover(a, b, seed=42),
    "frequency_crossover": frequency_crossover,
    "attention_weighted_crossover": lambda a, b: attention_weighted_crossover(
        a, b, temperature=1.0
    ),
    "uniform_crossover_with_mutation": lambda a, b: uniform_crossover_with_mutation(
        a, b, seed=42, sigma=0.05
    ),
    "multi_point_crossover": lambda a, b: multi_point_crossover(a, b, num_points=3, seed=42),
}
