import numpy as np
import pytest

from genome.synthesis import (
    N_PARENT_OPERATORS,
    average_n,
    frequency_crossover_n,
    recombine,
    single_point_crossover_n,
    uniform_crossover_n,
)


def test_average_n_three_parents():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    c = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    result = average_n([a, b, c])
    np.testing.assert_allclose(result, [4.0, 5.0, 6.0])
    assert result.dtype == np.float32


def test_uniform_crossover_n_each_dim_from_one_parent():
    parents = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32),
    ]
    result = uniform_crossover_n(parents, seed=42)
    # Each dim must be one of the parent values at that position
    for i, v in enumerate(result):
        assert v in (parents[0][i], parents[1][i], parents[2][i])


def test_uniform_crossover_n_deterministic_with_seed():
    parents = [np.array([1.0, 2.0], dtype=np.float32),
               np.array([10.0, 20.0], dtype=np.float32)]
    r1 = uniform_crossover_n(parents, seed=7)
    r2 = uniform_crossover_n(parents, seed=7)
    np.testing.assert_allclose(r1, r2)


def test_frequency_crossover_n_picks_largest_magnitude():
    parents = [
        np.array([10.0, 1.0, 5.0], dtype=np.float32),
        np.array([1.0, 20.0, 5.0], dtype=np.float32),
        np.array([2.0, 2.0, 100.0], dtype=np.float32),
    ]
    result = frequency_crossover_n(parents)
    assert result[0] == 10.0   # from parent 0
    assert result[1] == 20.0   # from parent 1
    assert result[2] == 100.0  # from parent 2


def test_single_parent_returns_self():
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_allclose(average_n([p]), p)
    np.testing.assert_allclose(single_point_crossover_n([p], seed=0), p)
    np.testing.assert_allclose(frequency_crossover_n([p]), p)


def test_recombine_dispatches():
    parents = [np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
    avg = recombine(parents, operator="simple_average")
    np.testing.assert_allclose(avg, [0.5, 0.5, 0.5, 0.5])


def test_recombine_unknown_operator_raises():
    parents = [np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
    with pytest.raises(ValueError):
        recombine(parents, operator="does-not-exist")


def test_all_registered_operators_callable():
    parents = [
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
    ]
    for name in N_PARENT_OPERATORS:
        out = recombine(parents, operator=name)
        assert out.shape == (4,), f"{name} returned wrong shape"
        assert out.dtype == np.float32, f"{name} returned wrong dtype"


def test_empty_parents_raises():
    with pytest.raises(ValueError):
        recombine([], operator="simple_average")


def test_mismatched_shapes_raises():
    with pytest.raises((ValueError, TypeError)):
        recombine(
            [np.zeros(4, dtype=np.float32), np.zeros(3, dtype=np.float32)],
            operator="simple_average",
        )


def test_single_point_crossover_empty_raises_valueerror():
    # Previously raised a confusing IndexError (skipped _stack).
    with pytest.raises(ValueError):
        recombine([], operator="single_point_crossover")


@pytest.mark.parametrize(
    "op",
    ["simple_average", "frequency_crossover", "uniform_crossover",
     "single_point_crossover", "uniform_crossover_with_mutation",
     "attention_weighted_crossover"],
)
def test_recombine_tolerates_unused_seed_kwarg(op):
    """A caller passing a uniform seed across operators must not crash on the
    seedless ones (simple_average / frequency_crossover)."""
    parents = [np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)]
    out = recombine(parents, operator=op, seed=7)
    assert out.shape == (4,)


def test_two_parent_operator_dim_guard_raises():
    from genome import operators as ops
    with pytest.raises(ValueError):
        ops.simple_average(
            np.ones(8, dtype=np.float32), np.ones(1, dtype=np.float32)
        )
