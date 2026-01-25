#!/usr/bin/env python3
"""Sum of squares of random numbers."""

import random
import sys
import time


def sum_of_squares(n: int, seed: int) -> float:
    """Pure Python version with loop."""
    random.seed(seed)
    total = 0.0
    for _ in range(n):
        x = random.random()
        total += x * x
    return total


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    n = 1_000_000
    iterations = 100

    start = time.perf_counter()
    for _ in range(iterations):
        result = sum_of_squares(n, seed)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / iterations

    print(f"Seed: {seed}, N: {n:,}, Iterations: {iterations}")
    print(f"Average time (Python): {avg_time:.4f} seconds")


if __name__ == "__main__":
    main()