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
