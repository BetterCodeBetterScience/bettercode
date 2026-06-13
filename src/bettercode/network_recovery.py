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
