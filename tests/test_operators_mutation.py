import numpy as np

from genome.operators import (
    gaussian_mutation,
    uniform_crossover,
    uniform_crossover_with_mutation,
)

A = np.arange(10, dtype=np.float32)
B = np.arange(100, 110, dtype=np.float32)


def test_gaussian_mutation_shape():
    result = gaussian_mutation(A, sigma=0.1, seed=0)
    assert result.shape == A.shape


def test_gaussian_mutation_noise_magnitude():
    result = gaussian_mutation(A, sigma=0.1, seed=0)
    diff = result - A
    # With sigma=0.1 and n=10, stdev of diff should be near 0.1
    assert 0.01 < np.std(diff) < 0.5


def test_gaussian_mutation_deterministic():
    r1 = gaussian_mutation(A, sigma=0.1, seed=42)
    r2 = gaussian_mutation(A, sigma=0.1, seed=42)
    assert np.allclose(r1, r2)


def test_gaussian_mutation_sigma_zero_returns_same():
    result = gaussian_mutation(A, sigma=0.0, seed=0)
    assert np.allclose(result, A)


def test_uniform_crossover_with_mutation():
    result = uniform_crossover_with_mutation(A, B, sigma=0.0, seed=0)
    # sigma=0 -> same as plain uniform_crossover
    expected = uniform_crossover(A, B, seed=0)
    assert np.allclose(result, expected)
