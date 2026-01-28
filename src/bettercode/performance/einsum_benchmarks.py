#!/usr/bin/env python3
"""Benchmark: einsum vs explicit NumPy operations."""

import time
import numpy as np

np.random.seed(42)


def benchmark(name: str, funcs: dict, n_runs: int = 100):
    """Run and compare multiple implementations."""
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print("=" * 60)
    
    results = {}
    values = {}
    
    for label, func in funcs.items():
        # Warmup
        func()
        
        # Time it
        start = time.perf_counter()
        for _ in range(n_runs):
            result = func()
        elapsed = time.perf_counter() - start
        
        results[label] = elapsed / n_runs * 1000  # ms
        values[label] = result
    
    # Verify results match
    ref_key = list(values.keys())[0]
    for label, val in values.items():
        if not np.allclose(val, values[ref_key]):
            print(f"WARNING: {label} result differs!")
    
    # Print results
    baseline = list(results.values())[0]
    for label, ms in results.items():
        speedup = baseline / ms
        print(f"  {label:30s}: {ms:8.3f} ms  ({speedup:5.2f}x)")


def main():
    print("Einsum vs Explicit NumPy Benchmarks")
    print("Arrays sized to show meaningful differences\n")
    
    # --- 1. Matrix multiplication ---
    A = np.random.rand(200, 300)
    B = np.random.rand(300, 400)
    
    benchmark("1. Matrix Multiplication (200x300 @ 300x400)", {
        "explicit (A @ B)": lambda: A @ B,
        "einsum": lambda: np.einsum('ij,jk->ik', A, B),
        "einsum optimized": lambda: np.einsum('ij,jk->ik', A, B, optimize=True),
    })
    
    # --- 2. Batch matrix multiplication ---
    batch = 50
    X = np.random.rand(batch, 64, 128)
    Y = np.random.rand(batch, 128, 64)
    
    benchmark("2. Batch Matrix Multiply (50 x 64x128 @ 128x64)", {
        "explicit (np.matmul)": lambda: np.matmul(X, Y),
        "explicit (loop)": lambda: np.stack([X[i] @ Y[i] for i in range(batch)]),
        "einsum": lambda: np.einsum('bij,bjk->bik', X, Y),
    })
    
    # --- 3. Outer product then sum ---
    a = np.random.rand(1000)
    b = np.random.rand(1000)
    
    benchmark("3. Sum of Outer Product (1000 x 1000)", {
        "explicit (outer + sum)": lambda: np.outer(a, b).sum(),
        "explicit (dot)": lambda: a.sum() * b.sum(),  # mathematically equivalent
        "einsum": lambda: np.einsum('i,j->', a, b),
    })
    
    # --- 4. Trace of matrix product ---
    M = np.random.rand(500, 500)
    N = np.random.rand(500, 500)
    
    benchmark("4. Trace of Product: tr(M @ N) (500x500)", {
        "explicit (M @ N then trace)": lambda: np.trace(M @ N),
        "explicit (sum of element-wise)": lambda: np.sum(M * N.T),
        "einsum": lambda: np.einsum('ij,ji->', M, N),
    })
    
    # --- 5. Tensor contraction (3D) ---
    T1 = np.random.rand(50, 60, 70)
    T2 = np.random.rand(70, 80)
    
    benchmark("5. Tensor Contraction (50x60x70 @ 70x80 -> 50x60x80)", {
        "explicit (reshape + matmul)": lambda: (T1.reshape(-1, 70) @ T2).reshape(50, 60, 80),
        "explicit (tensordot)": lambda: np.tensordot(T1, T2, axes=(2, 0)),
        "einsum": lambda: np.einsum('ijk,kl->ijl', T1, T2),
    }, n_runs=50)
    
    # --- 6. Multi-matrix chain ---
    P = np.random.rand(100, 120)
    Q = np.random.rand(120, 80)
    R = np.random.rand(80, 100)
    
    benchmark("6. Three Matrix Chain P @ Q @ R (100x120x80x100)", {
        "explicit ((P @ Q) @ R)": lambda: (P @ Q) @ R,
        "explicit (P @ (Q @ R))": lambda: P @ (Q @ R),
        "einsum": lambda: np.einsum('ij,jk,kl->il', P, Q, R),
        "einsum optimized": lambda: np.einsum('ij,jk,kl->il', P, Q, R, optimize=True),
    })
    
    # --- 7. Bilinear form x^T A y ---
    x = np.random.rand(500)
    A = np.random.rand(500, 500)
    y = np.random.rand(500)
    
    benchmark("7. Bilinear Form x^T @ A @ y (500)", {
        "explicit (x @ A @ y)": lambda: x @ A @ y,
        "einsum": lambda: np.einsum('i,ij,j->', x, A, y),
    })
    
    # --- 8. Diagonal of matrix product ---
    D1 = np.random.rand(400, 400)
    D2 = np.random.rand(400, 400)
    
    benchmark("8. Diagonal of Product diag(D1 @ D2) (400x400)", {
        "explicit (full product)": lambda: np.diag(D1 @ D2),
        "explicit (row-wise dot)": lambda: np.sum(D1 * D2.T, axis=1),
        "einsum": lambda: np.einsum('ij,ji->i', D1, D2),
    })
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    1. For simple ops (A @ B), built-ins are often faster
    2. Einsum shines when it avoids intermediate arrays
    3. optimize=True helps for multi-tensor contractions
    4. Trace/diagonal tricks: einsum avoids computing full matrix
    5. Batch operations: einsum is clean and competitive
    6. Readability: einsum makes complex contractions explicit
    """)


if __name__ == "__main__":
    main()