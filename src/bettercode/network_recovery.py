"""Recovering a known undirected network from simulated data.

Pedagogical example for the validation chapter: load the ecoli70 graph from
pgmpy, treat it as an undirected skeleton, encode that skeleton as the
sparsity pattern of a Gaussian precision matrix, simulate multivariate-normal
data with that precision, and recover the network by thresholding partial
correlations estimated from the sample. Because the precision sparsity *is*
the skeleton by construction, the partial-correlation ground truth is exactly
the skeleton (no moral-graph subtlety).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

Edge = tuple[str, str]


def build_covariance(
    edges: set[Edge], nodes: list[str], coupling: float
) -> np.ndarray:
    """Build a covariance matrix whose precision has the given sparsity pattern.

    Edges are encoded as off-diagonal nonzeros (value ``coupling``) of the
    precision matrix. The diagonal is set to ``sum(|off-diag|) + 0.1`` per row
    to guarantee strict diagonal dominance, which ensures positive-definiteness.
    The returned covariance is the inverse of this precision matrix.
    """
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    precision = np.zeros((n, n))
    for u, v in edges:
        i, j = node_idx[u], node_idx[v]
        precision[i, j] = precision[j, i] = coupling
    np.fill_diagonal(precision, np.abs(precision).sum(axis=1) + 0.1)
    return np.linalg.inv(precision)


def simulate_data(
    cov: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_samples`` from a zero-mean multivariate normal with covariance ``cov``."""
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n_samples)


def discover_edges(
    data: np.ndarray, nodes: list[str], threshold: float
) -> set[Edge]:
    """Recover edges by estimating partial correlations and thresholding their magnitude.

    Inverts the sample covariance to get an estimated precision matrix, then
    computes partial correlations ``-P_{ij} / sqrt(P_{ii} * P_{jj})``. Returns
    the set of node-name pairs (sorted lexicographically) whose partial
    correlation magnitude exceeds ``threshold``.
    """
    precision = np.linalg.inv(np.cov(data, rowvar=False))
    scale = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(scale, scale)
    discovered: set[Edge] = set()
    for i, j in combinations(range(len(nodes)), 2):
        if abs(partial_corr[i, j]) > threshold:
            discovered.add(tuple(sorted((nodes[i], nodes[j]))))
    return discovered


def score_recovery(
    true_edges: set[Edge], discovered_edges: set[Edge]
) -> dict[str, float]:
    """Score discovered edges against ground truth.

    Returns recall, precision, false-discovery rate, and F1. Edge-case conventions
    (chosen so ``precision + fdr == 1`` whenever both are defined and so trivial
    cases score as perfect):

    - both sets empty: recall=1, precision=1, fdr=0, f1=1
    - discovered empty, true non-empty: recall=0, precision=1, fdr=0, f1=0
    - true empty, discovered non-empty: recall=1, precision=0, fdr=1, f1=0
    """
    tp = len(true_edges & discovered_edges)
    fp = len(discovered_edges - true_edges)
    fn = len(true_edges - discovered_edges)

    if not true_edges and not discovered_edges:
        return {"recall": 1.0, "precision": 1.0, "fdr": 0.0, "f1": 1.0}

    recall = tp / len(true_edges) if true_edges else 1.0
    precision = tp / len(discovered_edges) if discovered_edges else 1.0
    fdr = fp / len(discovered_edges) if discovered_edges else 0.0
    f1_denom = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denom if f1_denom else 0.0
    return {"recall": recall, "precision": precision, "fdr": fdr, "f1": f1}


def load_ecoli_skeleton() -> tuple[set[Edge], list[str]]:
    """Load the ecoli70 Bayesian network from pgmpy and return its undirected skeleton.

    Returns:
        edges: set of (node1, node2) tuples with node1 < node2 lexicographically.
        nodes: sorted list of node names (used as a canonical index ordering).
    """
    from pgmpy.utils import get_example_model

    model = get_example_model("ecoli70")
    nodes = sorted(model.nodes())
    edges = {tuple(sorted((u, v))) for u, v in model.edges()}
    return edges, nodes


def run_signal_threshold_sweep(
    true_edges: set[Edge],
    nodes: list[str],
    signal_levels: Sequence[float],
    thresholds: Sequence[float],
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Score recovery across all (signal, threshold) combinations.

    For each signal level, the precision matrix is rebuilt with that coupling
    and fresh data is drawn. For each threshold, the same data is rescored.
    """
    rows = []
    for signal in signal_levels:
        cov = build_covariance(true_edges, nodes, coupling=signal)
        data = simulate_data(cov, n_samples, rng)
        for threshold in thresholds:
            discovered = discover_edges(data, nodes, threshold)
            stats = score_recovery(true_edges, discovered)
            rows.append({"signal": signal, "threshold": threshold, **stats})
    return pd.DataFrame(rows)


def plot_sweep_results(df: pd.DataFrame, save_path: Path) -> None:
    """Plot recall (solid) and FDR (dashed) vs. threshold, one color per signal level."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    signals = sorted(df["signal"].unique())
    palette = sns.color_palette("colorblind", n_colors=len(signals))

    fig, ax = plt.subplots(figsize=(8, 5))
    for color, signal in zip(palette, signals):
        sub = df[df["signal"] == signal].sort_values("threshold")
        ax.plot(
            sub["threshold"], sub["recall"],
            color=color, linestyle="-", marker="o",
            label=f"recall (signal={signal})",
        )
        ax.plot(
            sub["threshold"], sub["fdr"],
            color=color, linestyle="--", marker="x",
            label=f"FDR (signal={signal})",
        )
    ax.set_xlabel("partial-correlation threshold")
    ax.set_ylabel("rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
