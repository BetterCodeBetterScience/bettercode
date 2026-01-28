"""
Memory Profiling Example 1a: Memory Leak (BAD VERSION)

This script demonstrates a common memory leak pattern: accumulating all
intermediate results when you only need the final aggregate.

Usage:
    python -m memory_profiler memory_leak_bad.py

Install: pip install memory_profiler
"""

import numpy as np
import time

try:
    from memory_profiler import profile
except ImportError:
    print("ERROR: memory_profiler not installed!")
    print("Install with: pip install memory_profiler")
    exit(1)


@profile
def process_data_with_leak(n_iterations=10):
    all_results = []
    
    for i in range(n_iterations):
        data = np.random.randn(1000000)
        result = data ** 2
        
        all_results.append(result)
        
        time.sleep(0.1)
    
    final_sum = sum(np.sum(r) for r in all_results)
    return final_sum


def main():
    print("\n" + "=" * 80)
    print("MEMORY LEAK EXAMPLE (BAD VERSION)")
    print("=" * 80)
    print("\nThis version stores all intermediate results - memory leak!")
    print("Watch memory grow by ~80 MB (10 iterations × 8 MB each)")
    print("=" * 80)
    
    result = process_data_with_leak(n_iterations=10)
    print(f"\nResult: {result:.2f}")
    
    print("\n" + "=" * 80)
    print("WHAT HAPPENED")
    print("=" * 80)
    print("""
Memory grew continuously because:
- all_results.append(result) stores every array
- 10 iterations × 8 MB per array = ~80 MB accumulated
- We only needed the final sum, not all arrays!

Next, run the GOOD version to see the fix:
    python -m memory_profiler memory_leak_good.py
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
