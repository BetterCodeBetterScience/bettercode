"""
Memory Profiling Example 3a: Unnecessary Copies (BAD VERSION)

This script demonstrates how making unnecessary copies of data wastes memory.

Usage:
    python -m memory_profiler unnecessary_copies_bad.py

Install: pip install memory_profiler
"""

import numpy as np

try:
    from memory_profiler import profile
except ImportError:
    print("ERROR: memory_profiler not installed!")
    print("Install with: pip install memory_profiler")
    exit(1)


@profile
def process_with_copies(n_rows=2000000):
    data = np.random.randn(n_rows, 5)
    
    data_copy1 = data.copy() 
    data_normalized = (data_copy1 - data_copy1.mean()) / data_copy1.std()
    
    data_copy2 = data_normalized.copy() 
    data_squared = data_copy2 ** 2
    
    data_copy3 = data_squared.copy() 
    result = data_copy3.sum(axis=1)
    
    return result


def main():
    print("\n" + "=" * 80)
    print("UNNECESSARY COPIES EXAMPLE (BAD VERSION)")
    print("=" * 80)
    print("\nThis version makes unnecessary copies at each step!")
    print("Watch memory grow to ~320 MB (4 copies of 80 MB array)")
    print("=" * 80)
    
    result = process_with_copies(n_rows=2000000)
    print(f"\nProcessed {len(result):,} rows")
    
    print("\n" + "=" * 80)
    print("WHAT HAPPENED")
    print("=" * 80)
    print("""
Memory grew to ~320 MB because:
- data = np.random.randn(...) = 80 MB
- data_copy1 = data.copy() = another 80 MB
- data_copy2 = data_normalized.copy() = another 80 MB
- data_copy3 = data_squared.copy() = another 80 MB
- Peak memory: 4 × 80 MB = ~320 MB

These copies were unnecessary! We didn't need to preserve intermediate results.

Next, run the GOOD version to see the fix:
    python -m memory_profiler unnecessary_copies_good.py
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
