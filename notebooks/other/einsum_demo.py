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
# # Einstein Summation (einsum) Performance
#
# This notebook demonstrates when `np.einsum` helps vs. hurts performance.

# %%
import numpy as np
import time

np.random.seed(42)

def benchmark(funcs: dict, n_runs: int = 100) -> dict:
    """Time multiple implementations and return results in ms."""
    results = {}
    for label, func in funcs.items():
        func()  # warmup
        start = time.perf_counter()
        for _ in range(n_runs):
            func()
        results[label] = (time.perf_counter() - start) / n_runs * 1000
    
    baseline = list(results.values())[0]
    for label, ms in results.items():
        print(f"{label:35s}: {ms:7.3f} ms  ({baseline/ms:5.2f}x)")
    return results


# %% [markdown]
# ## Example 1: Matrix Multiplication (einsum is slower)
#
# For standard matrix multiplication, NumPy's `@` operator calls highly optimized BLAS libraries. Einsum adds overhead without benefit.
#
# $$C_{ik} = \sum_{j} A_{ij} B_{jk}$$

# %%
A = np.random.rand(200, 300)
B = np.random.rand(300, 400)

benchmark({
    "A @ B": lambda: A @ B,
    "einsum('ij,jk->ik', A, B)": lambda: np.einsum('ij,jk->ik', A, B),
})

# %% [markdown]
# **Takeaway:** For simple operations, use built-in operators.

# %% [markdown]
# ## Example 2: Sum of Outer Product (einsum is faster)
#
# Computing $\sum_{i,j} a_i b_j$ naively creates a large intermediate array. Einsum avoids this.
#
# $$\text{result} = \sum_{i} \sum_{j} a_i b_j$$

# %%
a = np.random.rand(1000)
b = np.random.rand(1000)

benchmark({
    "np.outer(a, b).sum()": lambda: np.outer(a, b).sum(),
    "einsum('i,j->', a, b)": lambda: np.einsum('i,j->', a, b),
})

# %% [markdown]
# **Why einsum wins:** The naive approach allocates a 1000×1000 intermediate array, then sums it. Einsum accumulates directly without the intermediate.

# %% [markdown]
# ## Example 3: Complex Chain (einsum + optimize shines)
#
# When contracting multiple tensors, the order of operations matters enormously. Einsum's `optimize=True` finds the best path.
#
# $$R_{im} = \sum_{j,k,l} W_{ij} X_{jk} Y_{kl} Z_{lm}$$

# %%
# Deliberately asymmetric sizes to make contraction order matter
W = np.random.rand(100, 5)
X = np.random.rand(5, 200)
Y = np.random.rand(200, 5)
Z = np.random.rand(5, 100)

benchmark({
    "((W @ X) @ Y) @ Z": lambda: ((W @ X) @ Y) @ Z,
    "W @ (X @ (Y @ Z))": lambda: W @ (X @ (Y @ Z)),
    "einsum (no optimize)": lambda: np.einsum('ij,jk,kl,lm->im', W, X, Y, Z),
    "einsum (optimize=True)": lambda: np.einsum('ij,jk,kl,lm->im', W, X, Y, Z, optimize=True),
}, n_runs=50)

# %%
# Verify all methods give the same result
result1 = ((W @ X) @ Y) @ Z
result2 = W @ (X @ (Y @ Z))
result3 = np.einsum('ij,jk,kl,lm->im', W, X, Y, Z)
result4 = np.einsum('ij,jk,kl,lm->im', W, X, Y, Z, optimize=True)

print("Results match:")
print(f"  Method 1 vs 2: {np.allclose(result1, result2)}")
print(f"  Method 1 vs 3: {np.allclose(result1, result3)}")
print(f"  Method 1 vs 4: {np.allclose(result1, result4)}")
print(f"  Method 3 vs 4: {np.allclose(result3, result4)}")

# %% [markdown]
# **Why order matters:**
#
# - Bad order creates large intermediate arrays
# - Good order keeps intermediates small
# - `optimize=True` uses dynamic programming to find the optimal contraction path
#
# You can inspect the chosen path:

# %%
path, info = np.einsum_path('ij,jk,kl,lm->im', W, X, Y, Z, optimize=True)
print(info)

# %% [markdown]
# ## Summary
#
# | Scenario | Recommendation |
# |----------|----------------|
# | Simple matrix ops | Use `@` operator |
# | Avoiding large intermediates | Use `einsum` |
# | Multi-tensor contractions | Use `einsum` with `optimize=True` |
