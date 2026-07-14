from genome.evaluate import run_evaluation


def test_run_evaluation_returns_results_per_operator():
    results = run_evaluation(limit_pairs=3)
    # One entry per operator
    assert "simple_average" in results
    assert "uniform_crossover" in results
    # Each entry has aggregated metrics
    avg_metrics = results["simple_average"]
    assert "precision@1" in avg_metrics
    assert "hit@5" in avg_metrics
    assert 0.0 <= avg_metrics["precision@1"] <= 1.0
