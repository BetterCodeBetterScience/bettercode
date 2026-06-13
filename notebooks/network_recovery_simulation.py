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
