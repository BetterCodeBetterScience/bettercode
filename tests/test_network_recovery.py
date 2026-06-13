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
