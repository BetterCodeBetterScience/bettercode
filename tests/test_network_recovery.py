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
