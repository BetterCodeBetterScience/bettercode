"""
Memory Profiling Example 4a: Array Growth (BAD VERSION)

This script demonstrates the inefficiency of growing arrays incrementally.

Usage:
    python -m memory_profiler array_growth_bad.py

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
def concatenate_with_append(n_iterations: int = 1000) -> np.ndarray:
    """Inefficiently build array by appending and concatenating.
    
    Parameters
    ----------
    n_iterations : int, default=1000
        Number of batches to generate and concatenate
    
    Returns
    -------
    np.ndarray
        Final concatenated array
    """
    results = []
    
    for i in range(n_iterations):
        batch = np.random.randn(1000)
        results.append(batch)
    
    final_array = np.concatenate(results)
    
    return final_array


def main() -> None:
    """Run the bad array growth example and explain the issue."""
    print("\n" + "=" * 80)
    print("ARRAY GROWTH EXAMPLE (BAD VERSION)")
    print("=" * 80)
    print("\nThis version grows array incrementally with append/concatenate!")
    print("Watch memory spike during concatenation as all arrays are copied")
    print("=" * 80)
    
    result = concatenate_with_append(n_iterations=1000)
    print(f"\nCreated array of size {len(result):,}")
    
    print("\n" + "=" * 80)
    print("WHAT HAPPENED")
    print("=" * 80)
    print("""
Memory spiked during concatenation because:
- results.append(batch) creates list of 1000 arrays
- np.concatenate(results) copies ALL 1000 arrays into new array
- Temporarily have both list and final array in memory
- Inefficient: data copied many times

Next, run the GOOD version to see the fix:
    python -m memory_profiler array_growth_good.py
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
