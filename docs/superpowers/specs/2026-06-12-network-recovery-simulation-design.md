# Network Recovery Simulation — Validation Chapter Code Replacement

**Date:** 2026-06-12
**Status:** Implemented (2026-06-13)

## Summary

Replace the existing causal-discovery code (pgmpy + tigramite + PCMCI) in the
"Simulating data from a model" subsection of the validation chapter with a
simpler partial-correlation-on-undirected-graph approach. The new code uses
the `ecoli70` graph from pgmpy as the source of structure, but treats it as
an undirected skeleton, generates multivariate normal data from a precision
matrix whose sparsity pattern matches the skeleton, and recovers the graph by
thresholding partial correlations.

## Motivation

The current code carries heavy dependencies (tigramite) and conceptual machinery
(structural causal processes, lag-1 autoregression, PCMCI) for what is, in the
chapter's narrative, an example of *simulating data from a known structure to
validate a recovery method*. Causal-discovery-from-timeseries is overkill and
distracting. A pure-numpy partial-correlation recovery makes the same
pedagogical point — known ground truth, simulated data, scored recovery,
parameter sweep showing expected behavior — with far less surface area.

The existing chapter already has a `TODO` flagging this:

> **TODO**: Simplify in favor of a simpler model that would only use Numpy?
> or acknowledge complex dependencies

## Scope

In scope:

1. New module `src/bettercode/network_recovery.py` implementing the algorithm.
2. Test module `tests/test_network_recovery.py` (TDD).
3. Create `notebooks/network_recovery_simulation.py` (jupytext percent format)
   as a standalone, self-contained notebook with inline equivalents of the new
   code. The existing `notebooks/network_recovery_simulation.py` (an untracked
   draft using a random network) is overwritten. The obsolete `## graphical
   modeling` section in `notebooks/simulation_examples.py` is left untouched
   for the author to delete manually — re-running that entire notebook is slow
   (PyMC sampling, OPTICS clustering), so cleanup is deferred.
4. Update `latex/book-validation.tex`: swap the six `lstlisting` blocks in the
   "Simulating data from a model" subsection for `lstinputlisting` blocks
   referencing the new module. Update the shell-output block's numbers.
5. Produce a written punch list of every prose/caption/figure-reference
   sentence in `book-validation.tex` that becomes inaccurate after the swap,
   for the author to revise manually (no LLM edits to narrative prose).

Out of scope:

- Editing any narrative prose in `book-validation.tex` (per the project rule).
- Touching any other section of `book-validation.tex` or any other chapter.
- Removing the `tigramite` dependency from `pyproject.toml` (the author may
  still want it referenced elsewhere; flag, don't remove).
- The "Simulating data based on existing data" subsection (uses SDV, unrelated).

## Design

### Algorithmic approach

The skeleton of `ecoli70` is encoded directly as the sparsity pattern of a
Gaussian precision matrix. Off-diagonal entries at edge positions are set to a
uniform value `coupling`; the diagonal of each row is set to
`sum(|off-diagonals in row|) + 0.1`, guaranteeing strict diagonal dominance
(and therefore positive-definiteness). The covariance is `inv(precision)`.
Multivariate-normal data is drawn from this covariance.

Recovery: invert the sample covariance to estimate the precision, convert to
partial correlations via `-P_{ij} / sqrt(P_{ii} P_{jj})`, threshold the
magnitude, and report the resulting edge set.

**Why this avoids moral-graph issues:** Because the precision matrix's
sparsity pattern *is* the skeleton by construction, the partial-correlation
ground truth is exactly the skeleton. The v-structure / moralization
subtleties that would arise if we generated data from the DAG via structural
equations do not apply here. This will be documented in a short module-level
docstring note.

### Module API

`src/bettercode/network_recovery.py`:

```python
Edge = tuple[str, str]  # node names, sorted lexicographically

def load_ecoli_skeleton() -> tuple[set[Edge], list[str]]:
    """Load pgmpy's ecoli70 model and return its undirected skeleton.

    Returns:
        edges: set of (node1, node2) with node1 < node2 lexicographically.
        nodes: list of node names in a fixed order (used to index the matrices).
    """

def build_covariance(
    edges: set[Edge], nodes: list[str], coupling: float
) -> np.ndarray:
    """Build a covariance matrix whose precision has the given sparsity pattern."""

def simulate_data(
    cov: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray: ...

def discover_edges(
    data: np.ndarray, nodes: list[str], threshold: float
) -> set[Edge]: ...

def score_recovery(
    true_edges: set[Edge], discovered_edges: set[Edge]
) -> dict[str, float]:
    """Return precision, recall, FDR, F1.

    Edge-case conventions (differ from the Downloads draft; "perfect" values
    listed alongside each):
      - true and discovered both empty:
          recall=1, precision=1, fdr=0, f1=1 (vacuously perfect)
      - discovered empty, true non-empty:
          recall=0, precision=1, fdr=0, f1=0
      - discovered non-empty, true empty:
          recall=1, precision=0, fdr=1, f1=0
    """

def run_signal_threshold_sweep(
    true_edges: set[Edge],
    nodes: list[str],
    signal_levels: Sequence[float],
    thresholds: Sequence[float],
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Cartesian sweep. One row per (signal, threshold) combination."""

def plot_sweep_results(df: pd.DataFrame, save_path: Path) -> None:
    """Recall and FDR vs. threshold, one line per signal level."""
```

Plus an `if __name__ == "__main__":` demo that runs the single recovery, prints
stats, runs the sweep, and saves the figure.

### Edge representation

`tuple[str, str]` with names sorted lexicographically (`tuple(sorted([u, v]))`).
This makes set membership work directly, gives a deterministic canonical form,
and avoids `frozenset` printing noise in docstrings/REPL.

### Defaults for the demo

- `n_samples = 500` (matches the existing example).
- `coupling = 0.4` for the single recovery.
- `threshold = 0.1` for the single recovery.
- Signal sweep: `[0.1, 0.2, 0.3, 0.4, 0.5]`.
- Threshold sweep: `[0.05, 0.10, 0.15, 0.20, 0.25]`.
- Seed: `42`.

These will be tuned after the first end-to-end run if the resulting figure
isn't clearly showing the expected pattern (recall rises with signal, FDR falls
as threshold tightens).

### Figure

Single panel. X-axis: threshold (linear). Y-axis: rate in [0, 1]. Two
line series per signal level: solid for recall, dashed for FDR. Colorblind-
friendly palette. Saved to
`book/book/images/network_recovery_performance.png`.

The existing figure file
`latex/files/causal_discovery_per-d0acdcca40112f89d07314020a18d03a.png` will be
removed in the same commit that adds the new figure. The `\includegraphics`
line in `book-validation.tex` must be updated to reference the new filename
— this is a mechanical, one-line change consequent on the rename, but because
it is not inside an `lstlisting` or `lstinputlisting` block, it falls outside
the LLM's allowed-edit zone per the latex/CLAUDE.md rule. It will be listed in
the punch list for the author to apply manually.

Note: the latex build pipeline hashes image filenames into the `latex/files/`
directory (e.g. `network_recovery_performance.png` in `book/book/images/`
becomes `network_recovery_performance-<hash>.png` in `latex/files/`). The
punch-list entry will specify the un-hashed name; the author will plug in the
final hashed path the build produces.

### Notebook update

Create a new standalone notebook `notebooks/network_recovery_simulation.py` in
jupytext percent format (`# %%` cell markers). It mirrors the new module: load,
build covariance, simulate, discover, score, sweep, plot. Reader-facing code is
self-contained — no `import` from `bettercode.network_recovery`. Notebook is
fully runnable in isolation (its own imports, its own `IMAGE_DIR` definition).
The existing `notebooks/simulation_examples.py` is NOT modified; its now-obsolete
`## graphical modeling` section is left for the author to delete manually
(re-running that notebook is slow due to PyMC + clustering, so cleanup is
deferred).

### Tex update

Six `lstlisting` blocks in `book-validation.tex`, currently lines ~186–323
(approximate; will be re-checked when editing). Each replaced with
`lstinputlisting[firstline=N, lastline=M]{../src/bettercode/network_recovery.py}`
using line ranges that map to the module's logical sections (load,
build_covariance, simulate + discover, score, sweep, plot). The shell-output
block (lines ~325–332) gets its numbers updated to the new code's actual output.

Path from `latex/book-validation.tex` to the module is `../src/bettercode/network_recovery.py`.

## Tests (TDD — written before implementation)

`tests/test_network_recovery.py`:

1. **`test_build_covariance_is_symmetric_positive_definite`** — small graph (4-node
   chain), check symmetry + all eigenvalues > 0.
2. **`test_build_covariance_sparsity_matches_edges`** — `inv(cov)` zeros are exactly
   at non-edge positions (up to numerical tolerance, e.g. abs < 1e-10).
3. **`test_discover_edges_recovers_chain`** — 4-node chain, coupling=0.5,
   n_samples=10000, threshold=0.1 → recovered edges == true edges.
4. **`test_score_recovery_basic`** — TP=2, FP=1, FN=1: recall=2/3, precision=2/3,
   FDR=1/3, F1=2/3.
5. **`test_score_recovery_empty_empty`** — both empty → all metrics = 1.0.
6. **`test_score_recovery_empty_discovered`** — true=2 edges, discovered=∅
   → recall=0, precision=1, FDR=0, F1=0.
7. **`test_score_recovery_empty_true`** — true=∅, discovered=2 edges → recall=1,
   precision=0, FDR=1, F1=0.
8. **`test_run_signal_threshold_sweep_shape`** — sweep returns a dataframe with
   `len(signals) * len(thresholds)` rows and the expected columns
   (`signal`, `threshold`, `recall`, `precision`, `fdr`, `f1`).
9. **`test_load_ecoli_skeleton`** — returns 46 nodes and 70 edges; every edge is
   a `tuple` of two strings in sorted order.

Test functions only (no classes); no type hints on test signatures (per project
CLAUDE.md). Pytest fixtures for shared rng / small graph.

## Prose-mismatch punch list (to be delivered as a comment on the PR / inline note)

After the code swap, these sentences/figures in `book-validation.tex` will be
inaccurate and need author revision:

- *"novel method for the discovery of causal relationships from timeseries
  data"* — no longer causal, no longer timeseries.
- *"directed acyclic graph, just like our workflow graphs in the previous
  chapter"* — now undirected.
- *"which has 46 nodes (representing individual genes) and 70 directed edges
  (representing causal relationships on gene expression between nodes)"* —
  edges are now undirected.
- *"we turn to the tigramite package, which is primarily focused on causal
  discovery from timeseries data"* — tigramite no longer used.
- The whole "convert the edge representation … to the link format required
  for tigramite" paragraph.
- *"PCMCI method described by \citep{Runge:2019aa}"* — no longer used.
- The Runge:2019aa citation will be unused (may still appear elsewhere; check).
- *"FDR-corrected p-values"* and *"q-value"* references — we now have a
  partial-correlation magnitude threshold, not a q-value.
- Figure caption *"true/false discovery rate as FDR increases"* — axes are
  now recall+FDR vs. partial-correlation threshold, not FDR threshold.
- The `\includegraphics{files/causal_discovery_per-...png}` line — must be
  updated to `\includegraphics{files/network_recovery_performance.png}` (or
  whatever path convention the latex build uses for `book/book/images`
  outputs). One-line mechanical change.
- *"two false relationships"* in the prose after the shell-output block —
  number will change with new code.
- The `TODO` line at the top of the subsection should be removed by the author
  once the prose is updated.

## Open questions / risks

1. **Default coupling/threshold values:** May need tuning to produce a sweep
   figure with a clearly visible signal × threshold interaction. Will adjust
   after first end-to-end run.
2. **pgmpy import time:** First import of `pgmpy.utils.get_example_model` is
   slow. Acceptable for a tutorial example; not optimized.
3. **Figure filename:** Keeping the existing hashed filename preserves the
   `\includegraphics` line and avoids prose edits, at the cost of a misleading
   name. Listed in the punch list.
