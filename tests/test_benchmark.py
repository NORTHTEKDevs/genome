
from genome.benchmark import (
    MetricWithCI,
    _bootstrap_ci,
    benchmark_diversity,
    benchmark_operators,
    save_benchmark,
)


def test_bootstrap_ci_returns_valid_range():
    vals = [0.2, 0.3, 0.25, 0.4, 0.35, 0.22, 0.28, 0.31, 0.29, 0.26]
    ci = _bootstrap_ci(vals, n_boot=200)
    assert isinstance(ci, MetricWithCI)
    assert 0.1 < ci.mean < 0.5
    assert ci.ci_low <= ci.mean <= ci.ci_high
    assert ci.n == len(vals)


def test_bootstrap_ci_empty_list():
    ci = _bootstrap_ci([])
    assert ci.mean == 0.0
    assert ci.n == 0


def test_benchmark_operators_smoke(tmp_path):
    from genome.dataset import load_parent_pairs

    # Only 5 pairs for speed
    pairs = load_parent_pairs()[:5]
    result = benchmark_operators(pairs=pairs, n_boot=50)
    assert "simple_average" in result
    assert "hit@3" in result["simple_average"]
    m = result["simple_average"]["hit@3"]
    assert 0.0 <= m.mean <= 1.0
    # CI brackets the mean
    assert m.ci_low <= m.mean <= m.ci_high

    out = tmp_path / "bench.json"
    save_benchmark(result, out)
    assert out.exists()


def test_benchmark_diversity_smoke():
    from genome.dataset import load_parent_pairs

    pairs = load_parent_pairs()[:3]
    result = benchmark_diversity(pairs=pairs, n_seeds=3)
    assert "uniform_crossover" in result
    d = result["uniform_crossover"]
    assert 0.0 <= d["mean_overlap"] <= 1.0
    assert 0.0 <= d["fraction_meaningfully_different"] <= 1.0
    assert d["n_pairs"] == 3
    assert d["n_seeds"] == 3
