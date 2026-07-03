# Network Recovery Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the causal-discovery (pgmpy + tigramite + PCMCI) example in the validation chapter with a simpler partial-correlation-on-undirected-graph recovery using the ecoli70 skeleton.

**Architecture:** A new pure-numpy module `src/bettercode/network_recovery.py` exposes six functions (load skeleton, build covariance, simulate, discover, score, sweep, plot). Test-driven from the inside out: small functions first, sweep + plot last, then integration via an `if __name__ == "__main__"` demo. Notebook gets inline replacement; chapter gets `lstinputlisting` swap; old figure removed and replaced; a punch list captures every prose sentence the author needs to revise.

**Tech Stack:** Python 3.13, numpy, scipy (not needed), pandas (sweep dataframe), pgmpy (graph loading only), matplotlib + seaborn (figure), pytest (TDD).

**Spec:** `docs/superpowers/specs/2026-06-12-network-recovery-simulation-design.md`

---

## File Structure

**Created:**
- `src/bettercode/network_recovery.py` — the module
- `tests/test_network_recovery.py` — tests
- `notebooks/network_recovery_simulation.py` — standalone reader-facing notebook (overwrites an existing untracked random-network draft)
- `docs/superpowers/plans/2026-06-12-network-recovery-prose-punch-list.md` — author's revision checklist (deliverable)
- `book/book/images/network_recovery_performance.png` — new figure

**Modified:**
- `latex/book-validation.tex` — swap six `lstlisting` blocks for `lstinputlisting`; update shell-output block numbers
- `pyproject.toml` — note (not remove) tigramite for author to decide later

**Left untouched (deferred for author):**
- `notebooks/simulation_examples.py` — the obsolete `## graphical modeling` section remains in place; deletion deferred because re-running the rest of the notebook (PyMC + clustering) is slow

**Deleted:**
- `latex/files/causal_discovery_per-d0acdcca40112f89d07314020a18d03a.png` — replaced by new figure

---

## Task 1: Project scaffolding & first failing test

**Files:**
- Create: `src/bettercode/network_recovery.py`
- Create: `tests/test_network_recovery.py`

- [ ] **Step 1: Create empty module file**

```python
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
```

Write that content to `src/bettercode/network_recovery.py`.

- [ ] **Step 2: Create test file scaffold**

```python
"""Tests for network_recovery module."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from bettercode import network_recovery as nr
```

Write that to `tests/test_network_recovery.py`.

- [ ] **Step 3: Verify pytest can collect the file (no tests yet, but file imports cleanly)**

Run: `uv run pytest tests/test_network_recovery.py --collect-only`
Expected: exit code 0, `no tests ran`.

- [ ] **Step 4: Commit**

```bash
git add src/bettercode/network_recovery.py tests/test_network_recovery.py
git commit -m "feat(network_recovery): scaffold module and test file"
```

---

## Task 2: `build_covariance` — failing test for symmetry + positive-definiteness

**Files:**
- Modify: `tests/test_network_recovery.py`
- Modify: `src/bettercode/network_recovery.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_network_recovery.py`:

```python
def test_build_covariance_is_symmetric_positive_definite():
    """A small chain graph should produce a symmetric PD covariance."""
    nodes = ["A", "B", "C", "D"]
    edges = {("A", "B"), ("B", "C"), ("C", "D")}
    cov = nr.build_covariance(edges, nodes, coupling=0.3)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    assert (np.linalg.eigvalsh(cov) > 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_network_recovery.py::test_build_covariance_is_symmetric_positive_definite -v`
Expected: FAIL with `AttributeError: module 'bettercode.network_recovery' has no attribute 'build_covariance'`.

- [ ] **Step 3: Implement `build_covariance`**

Append to `src/bettercode/network_recovery.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_network_recovery.py::test_build_covariance_is_symmetric_positive_definite -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_network_recovery.py src/bettercode/network_recovery.py
git commit -m "feat(network_recovery): build_covariance encodes edges into precision matrix"
```

---

## Task 3: `build_covariance` — sparsity pattern matches edges

**Files:**
- Modify: `tests/test_network_recovery.py`

This is a second test on the same function; no implementation change needed if it passes. Confirms the precision recovered from `inv(cov)` has zeros exactly at non-edge off-diagonal positions.

- [ ] **Step 1: Add test**

Append to `tests/test_network_recovery.py`:

```python
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
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/test_network_recovery.py::test_build_covariance_sparsity_matches_edges -v`
Expected: PASS (no implementation change).

- [ ] **Step 3: Commit**

```bash
git add tests/test_network_recovery.py
git commit -m "test(network_recovery): verify build_covariance sparsity pattern"
```

---

## Task 4: `simulate_data`

**Files:**
- Modify: `tests/test_network_recovery.py`
- Modify: `src/bettercode/network_recovery.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_network_recovery.py`:

```python
def test_simulate_data_shape_and_mean():
    """Drawing N samples from p-dim MVN gives (N, p) data with near-zero mean."""
    rng = np.random.default_rng(0)
    cov = np.eye(5)
    data = nr.simulate_data(cov, n_samples=10000, rng=rng)
    assert data.shape == (10000, 5)
    assert np.all(np.abs(data.mean(axis=0)) < 0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_network_recovery.py::test_simulate_data_shape_and_mean -v`
Expected: FAIL with `AttributeError: ... has no attribute 'simulate_data'`.

- [ ] **Step 3: Implement**

Append to `src/bettercode/network_recovery.py`:

```python
def simulate_data(
    cov: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``n_samples`` from a zero-mean multivariate normal with covariance ``cov``."""
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n_samples)
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_network_recovery.py::test_simulate_data_shape_and_mean -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_network_recovery.py src/bettercode/network_recovery.py
git commit -m "feat(network_recovery): simulate_data draws MVN samples"
```

---

## Task 5: `discover_edges` — failing test for chain recovery

**Files:**
- Modify: `tests/test_network_recovery.py`
- Modify: `src/bettercode/network_recovery.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_network_recovery.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_network_recovery.py::test_discover_edges_recovers_chain -v`
Expected: FAIL with `AttributeError: ... has no attribute 'discover_edges'`.

- [ ] **Step 3: Implement**

Append to `src/bettercode/network_recovery.py`:

```python
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
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_network_recovery.py::test_discover_edges_recovers_chain -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_network_recovery.py src/bettercode/network_recovery.py
git commit -m "feat(network_recovery): discover_edges via partial correlation thresholding"
```

---

## Task 6: `score_recovery` — basic case + edge cases

**Files:**
- Modify: `tests/test_network_recovery.py`
- Modify: `src/bettercode/network_recovery.py`

- [ ] **Step 1: Add four failing tests**

Append to `tests/test_network_recovery.py`:

```python
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
```

- [ ] **Step 2: Run all four to verify they fail**

Run: `uv run pytest tests/test_network_recovery.py -k score_recovery -v`
Expected: 4 FAIL (no `score_recovery` attribute).

- [ ] **Step 3: Implement**

Append to `src/bettercode/network_recovery.py`:

```python
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
```

- [ ] **Step 4: Run all four to verify they pass**

Run: `uv run pytest tests/test_network_recovery.py -k score_recovery -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_network_recovery.py src/bettercode/network_recovery.py
git commit -m "feat(network_recovery): score_recovery with documented edge-case conventions"
```

---

## Task 7: `load_ecoli_skeleton`

**Files:**
- Modify: `tests/test_network_recovery.py`
- Modify: `src/bettercode/network_recovery.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_network_recovery.py`:

```python
@pytest.mark.integration
def test_load_ecoli_skeleton():
    """ecoli70 has 46 nodes and 70 edges; each edge is a lexicographically sorted str tuple."""
    edges, nodes = nr.load_ecoli_skeleton()
    assert len(nodes) == 46
    assert len(edges) == 70
    for e in edges:
        assert isinstance(e, tuple) and len(e) == 2
        u, v = e
        assert isinstance(u, str) and isinstance(v, str)
        assert u < v  # sorted
```

The `integration` marker is already declared in `pyproject.toml`. This test hits pgmpy, which is slow on first import.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_network_recovery.py::test_load_ecoli_skeleton -v`
Expected: FAIL with `AttributeError: ... has no attribute 'load_ecoli_skeleton'`.

- [ ] **Step 3: Implement**

Append to `src/bettercode/network_recovery.py`:

```python
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
```

The `import` is local because pgmpy is heavy and not needed for the algorithmic tests.

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_network_recovery.py::test_load_ecoli_skeleton -v`
Expected: PASS. (First run will take several seconds for pgmpy import.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_network_recovery.py src/bettercode/network_recovery.py
git commit -m "feat(network_recovery): load_ecoli_skeleton via pgmpy"
```

---

## Task 8: `run_signal_threshold_sweep`

**Files:**
- Modify: `tests/test_network_recovery.py`
- Modify: `src/bettercode/network_recovery.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_network_recovery.py`:

```python
def test_run_signal_threshold_sweep_shape():
    """Sweep returns one row per (signal, threshold) with expected columns."""
    rng = np.random.default_rng(7)
    nodes = ["A", "B", "C", "D"]
    true_edges = {("A", "B"), ("B", "C"), ("C", "D")}
    signals = [0.2, 0.4]
    thresholds = [0.05, 0.1, 0.2]
    df = nr.run_signal_threshold_sweep(
        true_edges, nodes, signals, thresholds, n_samples=500, rng=rng
    )
    assert len(df) == 6
    assert set(df.columns) == {
        "signal", "threshold", "recall", "precision", "fdr", "f1",
    }
    assert sorted(df["signal"].unique().tolist()) == signals
    assert sorted(df["threshold"].unique().tolist()) == thresholds
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_network_recovery.py::test_run_signal_threshold_sweep_shape -v`
Expected: FAIL with `AttributeError: ... has no attribute 'run_signal_threshold_sweep'`.

- [ ] **Step 3: Implement**

Append to `src/bettercode/network_recovery.py`:

```python
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
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_network_recovery.py::test_run_signal_threshold_sweep_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_network_recovery.py src/bettercode/network_recovery.py
git commit -m "feat(network_recovery): run_signal_threshold_sweep"
```

---

## Task 9: `plot_sweep_results`

**Files:**
- Modify: `src/bettercode/network_recovery.py`

No unit test — figure generation is verified by the integration demo in Task 10.

- [ ] **Step 1: Implement**

Append to `src/bettercode/network_recovery.py`:

```python
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
```

Imports of `matplotlib` and `seaborn` are local so the algorithmic tests don't drag them in.

- [ ] **Step 2: Smoke test (manual)**

Run: `uv run python -c "
import numpy as np
from pathlib import Path
from bettercode import network_recovery as nr

rng = np.random.default_rng(42)
nodes = ['A', 'B', 'C', 'D']
edges = {('A', 'B'), ('B', 'C'), ('C', 'D')}
df = nr.run_signal_threshold_sweep(edges, nodes, [0.2, 0.4], [0.05, 0.1, 0.2], 500, rng)
nr.plot_sweep_results(df, Path('/tmp/test_sweep.png'))
print('saved')
"`
Expected output: `saved`. Confirm `/tmp/test_sweep.png` exists and renders.

- [ ] **Step 3: Commit**

```bash
git add src/bettercode/network_recovery.py
git commit -m "feat(network_recovery): plot_sweep_results figure"
```

---

## Task 10: Integration demo (`__main__` block) + tune defaults

**Files:**
- Modify: `src/bettercode/network_recovery.py`
- Create: `book/book/images/network_recovery_performance.png` (generated)

- [ ] **Step 1: Add the demo block**

Append to `src/bettercode/network_recovery.py`:

```python
def _demo() -> None:
    rng = np.random.default_rng(42)
    edges, nodes = load_ecoli_skeleton()
    n_samples = 500
    print(f"ecoli skeleton: {len(nodes)} nodes, {len(edges)} edges\n")

    # Single recovery at moderate signal and threshold.
    coupling, threshold = 0.4, 0.1
    cov = build_covariance(edges, nodes, coupling=coupling)
    data = simulate_data(cov, n_samples, rng)
    discovered = discover_edges(data, nodes, threshold=threshold)
    stats = score_recovery(edges, discovered)
    print(f"single run @ signal={coupling}, threshold={threshold}:")
    print(f"  {len(edges)} true edges, discovered {len(discovered)} edges")
    print(
        f"  recall={stats['recall']:.2%}  precision={stats['precision']:.2%}  "
        f"fdr={stats['fdr']:.2%}  f1={stats['f1']:.2%}\n"
    )

    # Sweep over signal × threshold.
    signals = [0.1, 0.2, 0.3, 0.4, 0.5]
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    df = run_signal_threshold_sweep(edges, nodes, signals, thresholds, n_samples, rng)
    print("sweep results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    image_dir = Path(__file__).resolve().parents[2] / "book" / "book" / "images"
    save_path = image_dir / "network_recovery_performance.png"
    plot_sweep_results(df, save_path)
    print(f"\nfigure saved to {save_path}")


if __name__ == "__main__":
    _demo()
```

- [ ] **Step 2: Run the demo**

Run: `uv run python -m bettercode.network_recovery`
Expected:
- Prints `ecoli skeleton: 46 nodes, 70 edges`.
- Prints a single-run summary line.
- Prints a 25-row sweep dataframe.
- Saves `book/book/images/network_recovery_performance.png`.

- [ ] **Step 3: Inspect the figure and the sweep numbers**

Open `book/book/images/network_recovery_performance.png` and verify:
1. As signal increases (lines colored by signal), recall (solid) curves stay near 1 longer as threshold grows.
2. As threshold tightens (rightward on x-axis), FDR (dashed) drops toward 0.
3. There is a visible separation between signal levels — if all five signal levels collapse to identical curves, the sweep grid needs adjustment.

If the figure does not clearly show the expected pattern, adjust the `signals` or `thresholds` lists in `_demo` and re-run until it does. Document the chosen values in the commit message.

- [ ] **Step 4: Commit**

```bash
git add src/bettercode/network_recovery.py book/book/images/network_recovery_performance.png
git commit -m "feat(network_recovery): demo runs end-to-end on ecoli skeleton"
```

---

## Task 11: Create standalone `notebooks/network_recovery_simulation.py`

**Files:**
- Create (overwriting any existing draft): `notebooks/network_recovery_simulation.py`

**Note:** The existing `notebooks/network_recovery_simulation.py` is an untracked draft
that demonstrates a random-network version of this idea. We overwrite its contents
entirely with the ecoli-based notebook below. `notebooks/simulation_examples.py` is
NOT modified; the obsolete `## graphical modeling` section there is left for the
author to delete manually (re-running that notebook is slow due to PyMC + clustering).

- [ ] **Step 1: Confirm the target file exists (as an untracked draft) and is safe to overwrite**

Run: `/bin/ls notebooks/network_recovery_simulation.py && git status --short notebooks/network_recovery_simulation.py`
Expected: file exists; `git status` shows it as `??` (untracked) — safe to overwrite.

- [ ] **Step 2: Write the full notebook contents**

Overwrite `notebooks/network_recovery_simulation.py` with the following.
This is the complete file — replace existing content in its entirety:

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bettercode
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Network recovery via partial correlation
#
# - load the ecoli70 graph from pgmpy and treat it as an undirected skeleton
# - encode that skeleton as the sparsity pattern of a Gaussian precision matrix
# - simulate multivariate-normal data from the implied covariance
# - recover the network by thresholding partial correlations from the sample
# - compare the recovered edges to the true skeleton

# %%
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pgmpy.utils import get_example_model

IMAGE_DIR = Path('../book/book/images')

# Load the ecoli70 model and extract its undirected skeleton.
ecoli_model = get_example_model('ecoli70')
nodes = sorted(ecoli_model.nodes())
true_edges = {tuple(sorted((u, v))) for u, v in ecoli_model.edges()}
print(f'{len(nodes)} nodes, {len(true_edges)} edges')


# %%
def build_covariance(edges, nodes, coupling):
    """Encode edges as the sparsity pattern of a precision matrix; return its inverse."""
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    precision = np.zeros((n, n))
    for u, v in edges:
        i, j = node_idx[u], node_idx[v]
        precision[i, j] = precision[j, i] = coupling
    np.fill_diagonal(precision, np.abs(precision).sum(axis=1) + 0.1)
    return np.linalg.inv(precision)


def simulate_data(cov, n_samples, rng):
    return rng.multivariate_normal(np.zeros(cov.shape[0]), cov, size=n_samples)


def discover_edges(data, nodes, threshold):
    """Estimate partial correlations from the sample and threshold their magnitude."""
    precision = np.linalg.inv(np.cov(data, rowvar=False))
    scale = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(scale, scale)
    discovered = set()
    for i, j in combinations(range(len(nodes)), 2):
        if abs(partial_corr[i, j]) > threshold:
            discovered.add(tuple(sorted((nodes[i], nodes[j]))))
    return discovered


def score_recovery(true_edges, discovered_edges):
    tp = len(true_edges & discovered_edges)
    fp = len(discovered_edges - true_edges)
    fn = len(true_edges - discovered_edges)
    if not true_edges and not discovered_edges:
        return {'recall': 1.0, 'precision': 1.0, 'fdr': 0.0, 'f1': 1.0}
    recall = tp / len(true_edges) if true_edges else 1.0
    precision = tp / len(discovered_edges) if discovered_edges else 1.0
    fdr = fp / len(discovered_edges) if discovered_edges else 0.0
    f1_denom = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denom if f1_denom else 0.0
    return {'recall': recall, 'precision': precision, 'fdr': fdr, 'f1': f1}


# %%
# Single recovery at moderate signal and threshold.
rng = np.random.default_rng(42)
n_samples = 500
coupling, threshold = 0.4, 0.1

cov = build_covariance(true_edges, nodes, coupling=coupling)
data = simulate_data(cov, n_samples, rng)
discovered = discover_edges(data, nodes, threshold=threshold)
stats = score_recovery(true_edges, discovered)

print(f'{len(true_edges)} true edges')
print(f'discovered {len(discovered)} edges')
print(f"Recall: {stats['recall']:.2%}")
print(f"Precision: {stats['precision']:.2%}")
print(f"False Discovery Rate: {stats['fdr']:.2%}")
print(f"F1 Score: {stats['f1']:.2%}")


# %%
# Sweep over signal strength and threshold.
signals = [0.1, 0.2, 0.3, 0.4, 0.5]
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
rows = []
for signal in signals:
    cov = build_covariance(true_edges, nodes, coupling=signal)
    data = simulate_data(cov, n_samples, rng)
    for threshold in thresholds:
        discovered = discover_edges(data, nodes, threshold)
        stats = score_recovery(true_edges, discovered)
        rows.append({'signal': signal, 'threshold': threshold, **stats})
performance_df = pd.DataFrame(rows)


# %%
# Plot recall (solid) and FDR (dashed) vs. threshold, one color per signal level.
colors = sns.color_palette('colorblind', n_colors=len(signals))
plt.figure(figsize=(8, 5))
for color, signal in zip(colors, signals):
    sub = performance_df[performance_df['signal'] == signal].sort_values('threshold')
    plt.plot(sub['threshold'], sub['recall'],
             color=color, linestyle='-', marker='o',
             label=f'recall (signal={signal})')
    plt.plot(sub['threshold'], sub['fdr'],
             color=color, linestyle='--', marker='x',
             label=f'FDR (signal={signal})')
plt.xlabel('partial-correlation threshold')
plt.ylabel('rate')
plt.ylim(-0.02, 1.02)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(IMAGE_DIR / 'network_recovery_performance.png', dpi=150, bbox_inches='tight')
plt.show()
```

The notebook is self-contained: it imports its own `sns`, `plt`, `np`, `pd`, and defines its own `IMAGE_DIR` at the top. It does not depend on any other notebook.

- [ ] **Step 3: Validate the notebook parses as Python**

Run: `uv run python -c "import ast; ast.parse(open('notebooks/network_recovery_simulation.py').read()); print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Run the notebook end-to-end**

Run: `uv run python notebooks/network_recovery_simulation.py 2>&1 | tail -40`
Expected: completes without error; output contains the single-run summary, the sweep loop result, and the figure save. No earlier-section warnings (this notebook is standalone).

- [ ] **Step 5: Commit**

```bash
git add notebooks/network_recovery_simulation.py
git commit -m "feat(notebook): standalone network-recovery simulation notebook"
```

---

## Task 12: Update `latex/book-validation.tex` — swap lstlisting to lstinputlisting

**Files:**
- Modify: `latex/book-validation.tex` (only `lstlisting` / `lstinputlisting` / numeric shell output blocks within `\subsection{Simulating data from a model}`)

The `latex/CLAUDE.md` rule: only contents of `lstlisting` blocks and `\lstinputlisting` options may be modified. Six code listings and one shell-output listing fall in scope; the surrounding prose is untouched.

- [ ] **Step 1: Locate the line ranges in `network_recovery.py` for each block**

Run: `uv run grep -n "^def \|^if __name__" src/bettercode/network_recovery.py`
Note the line numbers for each function definition; these will be the `firstline`/`lastline` values.

Map the six existing code blocks to the new file's functions:

| Existing block (purpose)                  | New file content                          |
|-------------------------------------------|-------------------------------------------|
| Load `ecoli_model` via pgmpy              | `load_ecoli_skeleton`                     |
| `generate_links_from_pgmpy_model`         | `build_covariance`                        |
| `generate_data` + invocation              | `simulate_data` + a sample call           |
| `run_pcmci`                                | `discover_edges`                          |
| `extract_discovered_links`                | (subsumed into `discover_edges`)          |
| `get_edge_stats`                          | `score_recovery`                          |

Note: the original chapter has six code blocks plus the sweep block. With the new module, the six blocks collapse to **five** logical code segments (load, build, simulate+discover, score, sweep+plot). The sixth block in the current chapter (the `extract_discovered_links` step) has no equivalent and should be deleted as part of this task.

- [ ] **Step 2: Replace block #1 — pgmpy load**

Find the block beginning with `from pgmpy.utils import get_example_model` (currently lines ~186–196 of `book-validation.tex`). Replace with:

```latex
\begin{lstlisting}[style=Python]
\end{lstlisting}
```

then replace with:

```latex
\lstinputlisting[style=Python, firstline=L1, lastline=L2]{../src/bettercode/network_recovery.py}
```

where `L1`/`L2` cover the `load_ecoli_skeleton` function in the module (use the grep output from Step 1).

Use Edit with `old_string` = the entire existing `\begin{lstlisting}[style=Python] ... \end{lstlisting}` block and `new_string` = the `\lstinputlisting` line.

- [ ] **Step 3: Replace block #2 — `generate_links_from_pgmpy_model`**

Similarly replace the block defining `generate_links_from_pgmpy_model` with `\lstinputlisting` for `build_covariance`.

- [ ] **Step 4: Replace block #3 — `generate_data`**

Replace with `\lstinputlisting` covering `simulate_data`.

- [ ] **Step 5: Replace block #4 — `run_pcmci`**

Replace with `\lstinputlisting` covering `discover_edges`.

- [ ] **Step 6: Delete block #5 — `extract_discovered_links`**

This block has no replacement (the new `discover_edges` returns the discovered set directly). Delete the entire `\begin{lstlisting}[style=Python] ... \end{lstlisting}` block. The surrounding prose ("First we need to extract…") will be flagged in the punch list as needing rewrite.

- [ ] **Step 7: Replace block #6 — `get_edge_stats`**

Replace with `\lstinputlisting` covering `score_recovery`.

- [ ] **Step 8: Replace block #7 — sweep loop**

Replace with `\lstinputlisting` covering `run_signal_threshold_sweep` (and optionally `plot_sweep_results`, depending on whether the chapter wants the plotting code visible).

- [ ] **Step 9: Update the shell-output block**

The existing block (lines ~325–332) reads:

```latex
\begin{lstlisting}[style=shell]
70 true edges
discovered 87 edges
True Positive Rate (Recall): 100.00%
Precision: 80.46%
False Discovery Rate: 19.54%
F1 Score: 89.17%
\end{lstlisting}
```

Run the demo to get the actual new numbers:

Run: `uv run python -m bettercode.network_recovery 2>&1 | grep -A 1 "single run"`

Replace the shell-output block with the actual numbers, keeping the same six-line format (true-edges count, discovered count, recall, precision, FDR, F1).

- [ ] **Step 10: Build the book to verify no LaTeX errors**

Run: `cd latex && make 2>&1 | tail -30`
Expected: no `Undefined control sequence` / `File not found` errors. PDF builds.

- [ ] **Step 11: Commit**

```bash
git add latex/book-validation.tex
git commit -m "refactor(book): swap validation chapter listings to lstinputlisting

Six lstlisting blocks now reference src/bettercode/network_recovery.py via
firstline/lastline ranges. Shell-output numbers updated to match the new
partial-correlation recovery results. One block (extract_discovered_links)
deleted as it has no equivalent in the new module."
```

---

## Task 13: Delete the old figure file

**Files:**
- Delete: `latex/files/causal_discovery_per-d0acdcca40112f89d07314020a18d03a.png`

- [ ] **Step 1: Confirm the file exists**

Run: `/bin/ls -la latex/files/causal_discovery_per-d0acdcca40112f89d07314020a18d03a.png`
Expected: file shown.

- [ ] **Step 2: Delete**

Run: `git rm latex/files/causal_discovery_per-d0acdcca40112f89d07314020a18d03a.png`

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(book): remove old causal_discovery figure (replaced by network_recovery_performance)"
```

---

## Task 14: Write the prose-mismatch punch list

**Files:**
- Create: `docs/superpowers/plans/2026-06-12-network-recovery-prose-punch-list.md`

- [ ] **Step 1: Write the punch list**

Write to `docs/superpowers/plans/2026-06-12-network-recovery-prose-punch-list.md`:

```markdown
# Network-recovery prose punch list

Code changes for the validation chapter's "Simulating data from a model" subsection
landed in this branch. The code now does partial-correlation recovery on an
undirected skeleton of the ecoli70 graph (no causal discovery, no timeseries, no
tigramite, no PCMCI). The following narrative needs author revision — the LLM
did not edit any prose per the latex/CLAUDE.md rule.

## Sentences and figure references that are now inaccurate

In `latex/book-validation.tex`, `\subsection{Simulating data from a model}`:

1. The `\textbf{TODO}` line at the top of the subsection can be removed.

2. **"Let's say that we have developed a tool that implements a novel method
   for the discovery of causal relationships from timeseries data."** — no
   longer about causal discovery or timeseries. Suggest: "Let's say we have
   developed a tool that recovers an undirected network from observational
   data."

3. **"generate data from a known causal graph (which is represented as a
   directed acyclic graph, just like our workflow graphs in the previous
   chapter)"** — now undirected. Suggest: "generate data from a known
   undirected network (the skeleton of a Bayesian network used in the gene
   expression literature)."

4. **"which has 46 nodes (representing individual genes) and 70 directed
   edges (representing causal relationships on gene expression between
   nodes)"** — edges are now treated as undirected. Suggest: "which has 46
   nodes (representing individual genes) and 70 edges (representing direct
   conditional dependencies between gene expression levels)."

5. **"Given this DAG, we then need to generate timeseries data for expression
   of each gene that reflect the causal relationships between the genes as
   well as the autocorrelation in gene expression within genes measured over
   time. For this, we turn to the *tigramite* package, which is primarily
   focused on causal discovery from timeseries data, but also includes a
   function that can generate timeseries data given a graphical model.
   However, the *tigramite* package requires a different representation of
   the graphical model than the one obtained from *pgmpy*, so we have to
   convert the edge representation from the original to the link format
   required for *tigramite*"** — entire paragraph needs replacement.
   Suggest: brief description of encoding the edge set as the sparsity
   pattern of a Gaussian precision matrix, with diagonal dominance for
   positive-definiteness, then sampling MVN data.

6. **"We can then create a function to take in the original model, convert
   it, and generate timeseries data for the model:"** — no longer
   timeseries. Suggest: "We then sample data from the multivariate normal
   distribution with this covariance:".

7. **"Now that we have the dataset we can test out our estimation method.
   Since I don't actually have a new method for causal estimation on
   timeseries, I will instead use the PCMCI method described by
   \citep{Runge:2019aa} and implemented in the *tigramite* package:"** —
   no longer using PCMCI or tigramite. Suggest: "Now that we have the
   dataset we can test out our estimation method, which estimates partial
   correlations between every pair of variables and thresholds their
   magnitudes to declare edges."

8. **"The results from this analysis include a list of all of the edges
   that were identified from the data using causal discovery, which we can
   summarize to determine how well the model performed. First we need to
   extract the links that were discovered from the results which pass our
   intended false discovery rate threshold:"** — `discover_edges` returns
   the set directly; there is no separate `extract_discovered_links` step
   and no q-value. Suggest: deleting this paragraph entirely (the listing
   it preceded is also deleted).

9. **"The results showed that the model performed quite well, detecting all
   of the true relationships and only two false relationships."** — exact
   numbers depend on the new code's run. Update after viewing the new
   shell-output block.

10. **"For example, we would expect better model performance with stronger
    signal, and we would expect fewer nodes identified when the p-value
    threshold is more stringent."** — no p-value threshold any more; it's
    a partial-correlation magnitude threshold. Suggest: "…fewer edges
    identified when the threshold is more stringent."

11. **"The results confirm that the model is performing as expected, with
    increasing recall as a function of increasing true signal and
    decreasing FDR threshold."** — threshold acts on partial-correlation
    magnitude now, not FDR. Suggest: "…with increasing recall as a function
    of increasing true signal, and decreasing FDR as the partial-correlation
    threshold tightens."

12. **Figure caption (`\caption[Plot of true/false discovery rate as FDR
    increases]{A plot of observed true positive rate (TPR) and false
    discovery rate (FDR) at increasing signal levels for varying FDR
    thresholds.}`)** — the threshold is no longer an FDR threshold.
    Suggest: "A plot of observed recall and false discovery rate (FDR) at
    increasing signal levels and varying partial-correlation thresholds."

13. **`\includegraphics{files/causal_discovery_per-d0acdcca40112f89d07314020a18d03a.png}`**
    must be updated to reference the new figure. The MyST/latex build
    rehashes filenames into `latex/files/`; the un-hashed name is
    `network_recovery_performance.png`. Replace with whatever the build
    produces (e.g. `files/network_recovery_performance-<newhash>.png`).

## References that may become orphaned

- `\citep{Runge:2019aa}` no longer appears in this subsection. Check whether
  it is cited anywhere else in the book; if not, the bibliography entry can
  be removed from `latex/references.bib`.
- `\citep{Schafer:2005aa}` is still relevant (the graph is from there).

## Dependencies that may become orphaned

- `tigramite>=5.2.9.4` in `pyproject.toml` is no longer used in this chapter.
  Confirm it's not used elsewhere before removing.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-06-12-network-recovery-prose-punch-list.md
git commit -m "docs: prose punch list for validation chapter author revision"
```

---

## Task 15: Final check — run full test suite

**Files:** none

- [ ] **Step 1: Run full pytest including integration markers**

Run: `uv run pytest tests/test_network_recovery.py -v`
Expected: all 9 tests pass.

- [ ] **Step 2: Run the full project test suite to confirm no regressions**

Run: `uv run pytest -x 2>&1 | tail -20`
Expected: all tests pass or fail only for unrelated reasons (note any new failures explicitly).

- [ ] **Step 3: Verify ruff & codespell still pass on the new file**

Run: `uv run ruff check src/bettercode/network_recovery.py tests/test_network_recovery.py`
Expected: no errors.

Run: `uv run codespell src/bettercode/network_recovery.py tests/test_network_recovery.py docs/superpowers/plans/2026-06-12-network-recovery-prose-punch-list.md`
Expected: no errors.

- [ ] **Step 4: Verify the latex build still works**

Run: `cd latex && make 2>&1 | tail -10`
Expected: PDF builds without errors.

- [ ] **Step 5: Final commit if needed**

If any cleanup commits are required from the checks above, commit them. Otherwise, the work is done.

---

## Self-review notes

- All seven items in the spec's `## Scope` "in scope" list have corresponding tasks (module: Tasks 1–10; tests: Tasks 2–8; notebook: Task 11; tex listings: Task 12; punch list: Task 14).
- The `score_recovery` edge-case conventions in the spec are implemented and tested explicitly (Task 6).
- The figure-rename consequence (`\includegraphics` line edit) is in the punch list (Task 14, item 13).
- The DAG → skeleton conversion uses `tuple(sorted((u, v)))` consistently across `load_ecoli_skeleton`, `discover_edges`, the notebook, and the tests.
- No placeholders remain.
