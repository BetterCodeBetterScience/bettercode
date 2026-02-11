#!/usr/bin/env python3
"""Sum of squares of random numbers."""

import sys
import time


class LCG:
    """Linear Congruential Generator matching Rust implementation (glibc parameters)."""
    
    def __init__(self, seed: int):
        self.state = seed
        
    def random(self) -> float:
        """Generate a random float in [0, 1) using LCG algorithm."""
        # LCG parameters (same as glibc and Rust implementation)
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        # Convert to float in [0, 1) - shift right 16 bits and divide by 2^48
        return (self.state >> 16) / (1 << 48)


def sum_of_squares(n: int, seed: int) -> float:
    """Pure Python version with loop using LCG RNG."""
    rng = LCG(seed)
    total = 0.0
    for _ in range(n):
        x = rng.random()
        total += x * x
    return total


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    n = 1_000_000
    iterations = 100

    # warm-up
    sum_of_squares(n, seed)
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = sum_of_squares(n, seed)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / iterations

    print(f"Seed: {seed}, N: {n:,}, Iterations: {iterations}")
    print(f"Average time (Python): {avg_time:.4f} seconds")


if __name__ == "__main__":
    main()