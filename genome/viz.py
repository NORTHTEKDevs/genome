"""Visualizations for GENOME evaluation results."""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_operator_bar_chart(
    results: dict[str, dict[str, float]],
    metric: str = "precision@5",
    output: Path | str = "results/bar_chart.png",
    title: str | None = None,
) -> None:
    """Save a horizontal bar chart of operators ranked by the given metric."""
    ranked = sorted(results.items(), key=lambda kv: kv[1].get(metric, 0.0))
    names = [k for k, _ in ranked]
    values = [v.get(metric, 0.0) for _, v in ranked]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.4)))
    ax.barh(names, values)
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1.0)
    ax.set_title(title or f"GENOME recombination operators: {metric}")
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_parent_hybrid_tsne(
    parents_a: np.ndarray,
    parents_b: np.ndarray,
    hybrids: np.ndarray,
    labels: list[str],
    output: Path | str = "results/tsne.png",
    title: str | None = None,
    perplexity: float | None = None,
) -> None:
    """t-SNE scatter of parent_a (blue), parent_b (orange), hybrid (green) per pair."""
    assert parents_a.shape == parents_b.shape == hybrids.shape
    n = parents_a.shape[0]
    all_vecs = np.concatenate([parents_a, parents_b, hybrids], axis=0)
    p = perplexity if perplexity is not None else min(30, max(5, (all_vecs.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=p, random_state=42, init="random")
    proj = tsne.fit_transform(all_vecs)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(proj[:n, 0], proj[:n, 1], c="tab:blue", label="parent A", s=60)
    ax.scatter(proj[n : 2 * n, 0], proj[n : 2 * n, 1], c="tab:orange", label="parent B", s=60)
    ax.scatter(proj[2 * n :, 0], proj[2 * n :, 1], c="tab:green", label="hybrid", s=80, marker="*")
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (proj[2 * n + i, 0], proj[2 * n + i, 1]), fontsize=7)
    ax.legend()
    ax.set_title(title or "Parents and hybrids (t-SNE)")
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
