#!/usr/bin/env python3
"""Sum of squares of random numbers."""

import sys
import time
import numpy as np


def sum_of_squares(n: int, seed: int) -> float:
    x = np.random.random(n)
    return np.dot(x, x)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    np.random.seed(seed)
    n = 1_000_000
    iterations = 100

    start = time.perf_counter()
    for _ in range(iterations):
        result = sum_of_squares(n, seed)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / iterations

    print(f"Seed: {seed}, N: {n:,}, Iterations: {iterations}")
    print(f"Average time (NumPy): {avg_time:.4f} seconds")


if __name__ == "__main__":
    main()