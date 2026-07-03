"""Recovering a known undirected network from simulated data.

Pedagogical example for the validation chapter: load the ecoli70 Bayesian
network from pgmpy, moralize it to obtain the undirected conditional
dependence graph (the moral graph), encode that graph as the sparsity pattern
of a Gaussian precision matrix, simulate multivariate-normal data with that
precision, and recover the network by thresholding partial correlations
estimated from the sample.

Why moralize? For a Gaussian DAG, the conditional independence structure is
given by the moral graph: two variables are conditionally independent given
all the others iff they are NOT adjacent in the moral graph. The moral graph
includes the DAG's skeleton plus undirected edges between any pair of nodes
that share a common child ("marry the parents"). Building the precision
matrix from the moral graph thus produces data whose true conditional
dependence structure matches the recovery target.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

Edge = tuple[str, str]


def build_covariance(
    edges: set[Edge], nodes: list[str], signal: float
) -> np.ndarray:
    """Build a covariance matrix with a known partial-correlation 
    signal.

    Constructs the precision matrix directly so that the 
    population partial correlation between every connected pair 
    of nodes equals ``signal``: the precision has 1 on the 
    diagonal and ``-signal`` at off-diagonal edge positions, since 
    the partial correlation between i and j is 
    ``-P_{ij} / sqrt(P_{ii} P_{jj})``. The matrix must be positive-
    definite, which constrains ``signal`` below 
    ``1 / lambda_max(adjacency)``; a clear error is raised if 
    violated. The returned covariance is the inverse.
    """
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    precision = np.eye(n)
    for u, v in edges:
        i, j = node_idx[u], node_idx[v]
        precision[i, j] = precision[j, i] = -signal
    try:
        np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"signal={signal} produces a precision matrix that is not "
            f"positive-definite for this graph; choose a smaller signal."
        ) from exc
    return np.linalg.inv(precision)


def simulate_data(
    cov: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_samples`` from a zero-mean multivariate normal with 
    covariance ``cov``."""
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n_samples)


def discover_edges(
    data: np.ndarray, nodes: list[str], q_threshold: float = 0.05
) -> set[Edge]:
    """Recover edges via an FDR-controlled partial-correlation test.

    For each pair of variables, the sample partial correlation is 
    computed from the inverse sample covariance and converted to a 
    p-value via the Fisher z-transform (under H0 of zero partial 
    correlation, conditioning on the other ``n_nodes - 2`` 
    variables, ``arctanh(r) * sqrt(n - p - 1)`` is approximately 
    N(0, 1)). The 2-sided p-values are then adjusted across
    all pairs using Benjamini-Hochberg, and an edge is declared 
    whenever the resulting q-value is at or below ``q_threshold``.
    """
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    n_samples, n_nodes = data.shape
    precision = np.linalg.inv(np.cov(data, rowvar=False))
    scale = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(scale, scale)

    pairs = list(combinations(range(n_nodes), 2))
    r_values = np.array([partial_corr[i, j] for i, j in pairs])
    z = np.arctanh(r_values) * np.sqrt(n_samples - n_nodes - 1)
    p_values = 2 * norm.sf(np.abs(z))

    reject, _, _, _ = multipletests(p_values, alpha=q_threshold, method="fdr_bh")
    return {
        tuple(sorted((nodes[i], nodes[j])))
        for (i, j), keep in zip(pairs, reject)
        if keep
    }


def score_recovery(
    true_edges: set[Edge], discovered_edges: set[Edge]
) -> dict[str, float]:
    """Score discovered edges against ground truth.

    Returns recall, precision, false-discovery rate, and F1. 
    Edge-case conventions (chosen so ``precision + fdr == 1`` 
    whenever both are defined and so trivial cases score as 
    perfect):

    - both sets empty: recall=1, precision=1, fdr=0, f1=1
    - discovered empty, true non-empty: recall=0, precision=1, 
        fdr=0, f1=0
    - true empty, discovered non-empty: recall=1, precision=0, 
        fdr=1, f1=0
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
    f1 = 2 * tp / f1_denom
    return {"recall": recall, "precision": precision, "fdr": fdr, "f1": f1}


def load_ecoli_moral_graph() -> tuple[set[Edge], list[str]]:
    """Load the ecoli70 Bayesian network from pgmpy and return its 
    moral graph.

    Moralization converts a DAG into the undirected graph that 
    represents the same conditional independence structure: each 
    pair of co-parents (nodes sharing a common child) gets 
    connected, then edge directions are dropped. For a Gaussian 
    DAG, the resulting moral graph is the sparsity pattern of 
    the precision matrix.

    Returns:
        edges: set of (node1, node2) tuples with node1 < node2
            lexicographically.
        nodes: sorted list of node names (used as a canonical
            index ordering).
    """
    from pgmpy.utils import get_example_model

    model = get_example_model("ecoli70")
    moral = model.moralize()
    nodes = sorted(model.nodes())
    edges = {tuple(sorted((u, v))) for u, v in moral.edges()}
    return edges, nodes


def run_signal_threshold_sweep(
    true_edges: set[Edge],
    nodes: list[str],
    signal_levels: Sequence[float],
    q_thresholds: Sequence[float],
    n_samples: int,
    n_runs: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Score recovery across all (signal, q-threshold) combinations.

    For each signal level (target partial-correlation magnitude), 
    the precision matrix is rebuilt and fresh data is drawn. For 
    each q-threshold, the same data is rescored using 
    ``discover_edges`` (FDR-controlled partial-correlation test).
    """
    rows = []
    for signal in signal_levels:
        for run in range(n_runs):
            cov = build_covariance(true_edges, nodes, signal=signal)
            data = simulate_data(cov, n_samples, rng)
            for q in q_thresholds:
                discovered = discover_edges(data, nodes, q_threshold=q)
                stats = score_recovery(true_edges, discovered)
                rows.append({"signal": signal, "q_threshold": q, **stats})
    return pd.DataFrame(rows)


def plot_sweep_results(df: pd.DataFrame, save_path: Path) -> None:
    """Plot recall (solid) and FDR (dashed) vs. signal level, one color per q-threshold.

    When the sweep contains multiple runs per (signal, q_threshold) cell, the
    line shows the mean across runs and the shaded band shows a 95% bootstrap
    confidence interval.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    long_df = df.melt(
        id_vars=["signal", "q_threshold"],
        value_vars=["recall", "fdr"],
        var_name="metric",
        value_name="rate",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=long_df,
        x="signal",
        y="rate",
        hue="q_threshold",
        style="metric",
        style_order=["recall", "fdr"],
        markers={"recall": "o", "fdr": "X"},
        dashes={"recall": "", "fdr": (3, 2)},
        palette="colorblind",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("signal (true partial-correlation magnitude)")
    ax.set_ylabel("rate (recall/FDR)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _demo() -> None:
    """Run an end-to-end demonstration of network recovery on the ecoli moral graph."""
    rng = np.random.default_rng(42)
    edges, nodes = load_ecoli_moral_graph()
    n_samples = 1000
    print(f"ecoli moral graph: {len(nodes)} nodes, {len(edges)} edges\n")

    # Single recovery at moderate signal and a 5% FDR threshold.
    signal, q_threshold = 0.12, 0.05
    cov = build_covariance(edges, nodes, signal=signal)
    data = simulate_data(cov, n_samples, rng)
    discovered = discover_edges(data, nodes, q_threshold=q_threshold)
    true_positive_edges = np.array([True if e in discovered else False for e in edges])
    false_positive_edges = np.array([True if e not in edges else False for e in discovered])
    stats = score_recovery(edges, discovered)
    print(f"single run @ signal={signal}, q_threshold={q_threshold}:")
    print(f"  {len(edges)} true edges, discovered {len(discovered)} edges")
    print(f"  {sum(true_positive_edges)} true positives, {sum(false_positive_edges)} false positives")
    print(
        f"  recall={stats['recall']:.2%}  precision={stats['precision']:.2%}  "
        f"fdr={stats['fdr']:.2%}  f1={stats['f1']:.2%}\n"
    )

    # Sweep over signal × q-threshold. Signals span below the noise floor
    # up to near the PD limit (~0.158 for the ecoli70 moral graph).
    signals = np.arange(0.04, 0.15, 0.02)
    q_thresholds = [1e-3, 1e-2, 5e-2]
    n_runs = 10
    df = run_signal_threshold_sweep(edges, nodes, signals, q_thresholds, n_samples, n_runs, rng)
    print("sweep results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    image_dir = Path(__file__).resolve().parents[2] / "latex" / "files"
    save_path = image_dir / "network_recovery_performance.png"
    plot_sweep_results(df, save_path)
    print(f"\nfigure saved to {save_path}")


if __name__ == "__main__":
    _demo()
