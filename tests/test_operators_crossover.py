import numpy as np

from genome.operators import (
    attention_weighted_crossover,
    frequency_crossover,
    multi_point_crossover,
    single_point_crossover,
    uniform_crossover,
)

A = np.arange(10, dtype=np.float32)
B = np.arange(100, 110, dtype=np.float32)


def test_single_point_crossover_shape():
    result = single_point_crossover(A, B, crossover_point=5)
    assert result.shape == A.shape


def test_single_point_crossover_takes_a_then_b():
    result = single_point_crossover(A, B, crossover_point=3)
    assert np.allclose(result[:3], A[:3])
    assert np.allclose(result[3:], B[3:])


def test_single_point_crossover_endpoints():
    result_start = single_point_crossover(A, B, crossover_point=0)
    assert np.allclose(result_start, B)
    result_end = single_point_crossover(A, B, crossover_point=A.shape[0])
    assert np.allclose(result_end, A)


def test_single_point_crossover_random_seed_deterministic():
    r1 = single_point_crossover(A, B, seed=7)
    r2 = single_point_crossover(A, B, seed=7)
    assert np.allclose(r1, r2)


def test_uniform_crossover_shape():
    result = uniform_crossover(A, B, seed=0)
    assert result.shape == A.shape


def test_uniform_crossover_each_dim_from_one_parent():
    result = uniform_crossover(A, B, seed=42)
    for i, v in enumerate(result):
        assert v == A[i] or v == B[i], f"dim {i} value {v} not from either parent"


def test_uniform_crossover_mask_prob_half_roughly_balanced():
    results = [uniform_crossover(A, B, seed=s, prob_a=0.5) for s in range(100)]
    matches_a = sum(np.sum(r == A) for r in results) / (100 * A.shape[0])
    assert 0.4 < matches_a < 0.6, f"expected ~0.5, got {matches_a}"


def test_uniform_crossover_prob_a_zero_returns_b():
    result = uniform_crossover(A, B, seed=0, prob_a=0.0)
    assert np.allclose(result, B)


def test_uniform_crossover_prob_a_one_returns_a():
    result = uniform_crossover(A, B, seed=0, prob_a=1.0)
    assert np.allclose(result, A)


def test_frequency_crossover_picks_larger_magnitude():
    # A has larger magnitude in dim 0, B has larger magnitude in dim 1
    a = np.array([10.0, 1.0, 5.0, 5.0], dtype=np.float32)
    b = np.array([1.0, 10.0, 5.0, 5.0], dtype=np.float32)
    result = frequency_crossover(a, b)
    assert result[0] == 10.0  # from a
    assert result[1] == 10.0  # from b
    # ties go to a (first arg) by convention
    assert result[2] == 5.0
    assert result[3] == 5.0


def test_frequency_crossover_shape():
    result = frequency_crossover(A, B)
    assert result.shape == A.shape


def test_attention_weighted_shape():
    result = attention_weighted_crossover(A, B, temperature=1.0)
    assert result.shape == A.shape


def test_attention_weighted_high_temp_approaches_average():
    # Very high temperature -> softmax is uniform -> result near average.
    # atol=1e-2 accounts for |b|*|a-b|/T float convergence at T=1e6 with magnitudes ~100.
    result = attention_weighted_crossover(A, B, temperature=1e6)
    avg = (A + B) / 2.0
    assert np.allclose(result, avg, atol=1e-2)


def test_attention_weighted_low_temp_approaches_frequency():
    # Very low temperature -> softmax is sharp -> result near frequency_crossover
    a = np.array([10.0, 1.0, 5.0, 5.0], dtype=np.float32)
    b = np.array([1.0, 10.0, 5.0, 5.0], dtype=np.float32)
    result = attention_weighted_crossover(a, b, temperature=1e-3)
    freq = frequency_crossover(a, b)
    # Low temp should be very close to frequency result
    assert np.allclose(result, freq, atol=0.5)


def test_multi_point_crossover_shape():
    result = multi_point_crossover(A, B, num_points=3, seed=0)
    assert result.shape == A.shape


def test_multi_point_crossover_uses_both_parents():
    result = multi_point_crossover(A, B, num_points=3, seed=7)
    matches_a = np.sum(result == A)
    matches_b = np.sum(result == B)
    assert matches_a + matches_b == A.shape[0]
    assert matches_a > 0
    assert matches_b > 0


def test_multi_point_crossover_zero_points_equals_one_parent():
    result = multi_point_crossover(A, B, num_points=0, seed=0)
    assert np.allclose(result, A) or np.allclose(result, B)
