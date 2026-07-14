import numpy as np

from genome.operators import concat_project, simple_average, weighted_sum

A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
B = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)


def test_simple_average():
    result = simple_average(A, B)
    expected = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    assert np.allclose(result, expected)
    assert result.shape == A.shape


def test_weighted_sum_alpha_half():
    result = weighted_sum(A, B, alpha=0.5)
    assert np.allclose(result, simple_average(A, B))


def test_weighted_sum_alpha_zero():
    result = weighted_sum(A, B, alpha=0.0)
    assert np.allclose(result, B)


def test_weighted_sum_alpha_one():
    result = weighted_sum(A, B, alpha=1.0)
    assert np.allclose(result, A)


def test_concat_project_preserves_dim():
    result = concat_project(A, B, projection_seed=42)
    assert result.shape == A.shape
