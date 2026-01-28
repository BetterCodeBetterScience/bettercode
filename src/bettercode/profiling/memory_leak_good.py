"""
Memory Profiling Example 1b: Memory Leak (GOOD VERSION)

This script demonstrates the corrected version: only accumulate what you need.

Usage:
    python -m memory_profiler memory_leak_good.py

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
def process_data_efficiently(n_iterations=10):
    running_sum = 0.0
    
    for i in range(n_iterations):
        data = np.random.randn(1000000) 
        result = data ** 2
        
        running_sum += np.sum(result)
        
        time.sleep(0.1)
    
    return running_sum


def main():
    print("\n" + "=" * 80)
    print("MEMORY LEAK EXAMPLE (GOOD VERSION)")
    print("=" * 80)
    print("\nThis version only accumulates the running sum - constant memory!")
    print("Watch memory oscillate but stay relatively constant")
    print("=" * 80)
    
    result = process_data_efficiently(n_iterations=10)
    print(f"\nResult: {result:.2f}")
    
    print("\n" + "=" * 80)
    print("WHAT HAPPENED")
    print("=" * 80)
    print("""
Memory stayed constant because:
- running_sum += np.sum(result) only keeps the aggregate
- Each result array is garbage collected after use
- No memory leak - just temporary ~8 MB oscillations

KEY INSIGHT:
- BAD: all_results.append(result) - stores everything
- GOOD: running_sum += np.sum(result) - accumulates only needed values

This pattern is common when:
- Processing data in a loop
- You only need aggregate statistics (sum, mean, max, etc.)
- But you accidentally store all intermediate arrays

The fix: Only accumulate what you actually need!
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
