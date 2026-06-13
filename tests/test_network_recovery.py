"""Tests for network_recovery module."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from bettercode import network_recovery as nr


def test_build_covariance_is_symmetric_positive_definite():
    """A small chain graph should produce a symmetric PD covariance."""
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "B"), ("B", "C"), ("C", "D")}
    cov = nr.build_covariance(edges, nodes, coupling=0.3)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    assert (np.linalg.eigvalsh(cov) > 0).all()


def test_build_covariance_sparsity_matches_edges():
    """inv(cov) should have nonzeros exactly at edge positions."""
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "B"), ("C", "D")}  # two disjoint edges
    cov = nr.build_covariance(edges, nodes, coupling=0.3)
    precision = np.linalg.inv(cov)
    # off-diagonal positions: edges are nonzero, non-edges are zero
    assert abs(precision[0, 1]) > 0.1  # A-B edge
    assert abs(precision[2, 3]) > 0.1  # C-D edge
    assert abs(precision[0, 2]) < 1e-10  # A-C non-edge
    assert abs(precision[0, 3]) < 1e-10  # A-D non-edge
    assert abs(precision[1, 2]) < 1e-10  # B-C non-edge
    assert abs(precision[1, 3]) < 1e-10  # B-D non-edge


def test_simulate_data_shape_and_mean():
    """Drawing N samples from p-dim MVN gives (N, p) data with near-zero mean."""
    rng = np.random.default_rng(0)
    cov = np.eye(5)
    data = nr.simulate_data(cov, n_samples=10000, rng=rng)
    assert data.shape == (10000, 5)
    assert np.all(np.abs(data.mean(axis=0)) < 0.1)


def test_discover_edges_recovers_chain():
    """With strong coupling and many samples, partial-correlation thresholding
    recovers the true edge set of a 4-node chain."""
    rng = np.random.default_rng(123)
    nodes = ["A", "B", "C", "D"]
    true_edges = {("A", "B"), ("B", "C"), ("C", "D")}
    cov = nr.build_covariance(true_edges, nodes, coupling=0.5)
    data = nr.simulate_data(cov, n_samples=10000, rng=rng)
    discovered = nr.discover_edges(data, nodes, threshold=0.1)
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
