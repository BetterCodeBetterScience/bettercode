"""Tests for network_recovery module."""

import numpy as np
import pytest

from bettercode import network_recovery as nr


def test_build_covariance_is_symmetric_positive_definite():
    """A small chain graph should produce a symmetric PD covariance."""
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "B"), ("B", "C"), ("C", "D")}
    cov = nr.build_covariance(edges, nodes, signal=0.3)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    assert (np.linalg.eigvalsh(cov) > 0).all()


def test_build_covariance_partial_correlations_match_signal():
    """For connected pairs, the population partial correlation equals ``signal``."""
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "B"), ("B", "C"), ("C", "D")}
    signal = 0.3
    cov = nr.build_covariance(edges, nodes, signal=signal)
    precision = np.linalg.inv(cov)
    scale = np.sqrt(np.diag(precision))
    pcor = -precision / np.outer(scale, scale)
    # A-B, B-C, C-D are edges → partial correlation == signal
    assert pcor[0, 1] == pytest.approx(signal)
    assert pcor[1, 2] == pytest.approx(signal)
    assert pcor[2, 3] == pytest.approx(signal)
    # non-edges → partial correlation == 0
    assert abs(pcor[0, 2]) < 1e-12
    assert abs(pcor[0, 3]) < 1e-12
    assert abs(pcor[1, 3]) < 1e-12


def test_build_covariance_rejects_non_pd_signal():
    """A signal too large for the graph raises a clear ValueError."""
    nodes = ["A", "B", "C"]
    edges = {("A", "B"), ("B", "C"), ("A", "C")}  # 3-clique → lambda_max(A) = 2
    with pytest.raises(ValueError, match="not positive-definite"):
        nr.build_covariance(edges, nodes, signal=0.6)  # 0.6 > 1/2


def test_simulate_data_shape_and_mean():
    """Drawing N samples from p-dim MVN gives (N, p) data with near-zero mean."""
    rng = np.random.default_rng(0)
    cov = np.eye(5)
    data = nr.simulate_data(cov, n_samples=10000, rng=rng)
    assert data.shape == (10000, 5)
    assert np.all(np.abs(data.mean(axis=0)) < 0.1)


def test_discover_edges_recovers_chain():
    """With strong signal and many samples, FDR-controlled partial-correlation
    recovery returns the true edge set of a 4-node chain."""
    rng = np.random.default_rng(123)
    nodes = ["A", "B", "C", "D"]
    true_edges = {("A", "B"), ("B", "C"), ("C", "D")}
    cov = nr.build_covariance(true_edges, nodes, signal=0.5)
    data = nr.simulate_data(cov, n_samples=10000, rng=rng)
    discovered = nr.discover_edges(data, nodes, q_threshold=0.05)
    assert discovered == true_edges


def test_score_recovery_basic():
    """TP=2, FP=1, FN=1 → recall=2/3, precision=2/3, fdr=1/3, f1=2/3."""
    true_edges = {("A", "B"), ("B", "C"), ("C", "D")}
    discovered = {("A", "B"), ("B", "C"), ("D", "E")}
    s = nr.score_recovery(true_edges, discovered)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["fdr"] == pytest.approx(1 / 3)
    assert s["f1"] == pytest.approx(2 / 3)


def test_score_recovery_empty_empty():
    """Empty true & empty discovered → vacuously perfect."""
    s = nr.score_recovery(set(), set())
    assert s == {"recall": 1.0, "precision": 1.0, "fdr": 0.0, "f1": 1.0}


def test_score_recovery_empty_discovered():
    """True edges exist, discovered nothing → recall=0, precision=1, fdr=0, f1=0."""
    true_edges = {("A", "B"), ("B", "C")}
    s = nr.score_recovery(true_edges, set())
    assert s == {"recall": 0.0, "precision": 1.0, "fdr": 0.0, "f1": 0.0}


def test_score_recovery_empty_true():
    """No true edges, discovered some → recall=1, precision=0, fdr=1, f1=0."""
    discovered = {("A", "B"), ("B", "C")}
    s = nr.score_recovery(set(), discovered)
    assert s == {"recall": 1.0, "precision": 0.0, "fdr": 1.0, "f1": 0.0}


@pytest.mark.integration
def test_load_ecoli_moral_graph():
    """ecoli70 has 46 nodes; its moral graph adds co-parent edges (84 total)."""
    edges, nodes = nr.load_ecoli_moral_graph()
    assert len(nodes) == 46
    assert len(edges) == 84  # 70 skeleton edges + 14 co-parent edges
    for e in edges:
        assert isinstance(e, tuple) and len(e) == 2
        u, v = e
        assert isinstance(u, str) and isinstance(v, str)
        assert u < v  # sorted


def test_run_signal_threshold_sweep_shape():
    """Sweep returns one row per (signal, q_threshold) with expected columns."""
    rng = np.random.default_rng(7)
    nodes = ["A", "B", "C", "D"]
    true_edges = {("A", "B"), ("B", "C"), ("C", "D")}
    signals = [0.2, 0.4]
    q_thresholds = [0.001, 0.01, 0.05]
    n_runs = 3
    df = nr.run_signal_threshold_sweep(
        true_edges, nodes, signals, q_thresholds, n_samples=500, n_runs=n_runs, rng=rng
    )
    # One row per (signal, run, q_threshold)
    assert len(df) == len(signals) * n_runs * len(q_thresholds)
    assert set(df.columns) == {
        "signal", "q_threshold", "recall", "precision", "fdr", "f1",
    }
    assert sorted(df["signal"].unique().tolist()) == signals
    assert sorted(df["q_threshold"].unique().tolist()) == q_thresholds
