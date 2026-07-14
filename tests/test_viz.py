import matplotlib

matplotlib.use("Agg")
import numpy as np

from genome.viz import plot_operator_bar_chart, plot_parent_hybrid_tsne


def test_plot_operator_bar_chart_saves_file(tmp_path):
    results = {
        "simple_average": {"precision@5": 0.2},
        "uniform_crossover": {"precision@5": 0.45},
    }
    out = tmp_path / "plot.png"
    plot_operator_bar_chart(results, metric="precision@5", output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_tsne_saves_file(tmp_path):
    rng = np.random.default_rng(0)
    # 3 pairs, 384 dims. Produce parent_a, parent_b, and hybrid vectors.
    parents_a = rng.standard_normal((3, 384)).astype(np.float32)
    parents_b = rng.standard_normal((3, 384)).astype(np.float32)
    hybrids = (parents_a + parents_b) / 2.0

    out = tmp_path / "tsne.png"
    plot_parent_hybrid_tsne(
        parents_a,
        parents_b,
        hybrids,
        labels=["pair_001", "pair_002", "pair_003"],
        output=out,
    )
    assert out.exists()
    assert out.stat().st_size > 0
