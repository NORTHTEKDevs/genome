import numpy as np

from genome.operators import OPERATORS


def test_registry_has_all_operators():
    expected = {
        "simple_average",
        "weighted_sum",
        "concat_project",
        "single_point_crossover",
        "uniform_crossover",
        "frequency_crossover",
        "attention_weighted_crossover",
        "uniform_crossover_with_mutation",
        "multi_point_crossover",
    }
    assert set(OPERATORS.keys()) == expected


def test_registry_operators_are_callable():
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([5, 6, 7, 8], dtype=np.float32)
    for name, op in OPERATORS.items():
        result = op(a, b)
        assert result.shape == a.shape, f"{name} returned wrong shape"
        assert result.dtype == np.float32, f"{name} returned wrong dtype"
