"""
Memory Profiling Example 2a: Chunked Processing (BAD VERSION)

This script demonstrates loading an entire large dataset into memory at once.

Usage:
    python -m memory_profiler chunked_processing_bad.py

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
def analyze_large_dataset_all_at_once(n_samples=5000000):
    data = np.random.randn(n_samples, 10)  # ~400 MB
    
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    
    result = {
        'mean': mean_vals,
        'std': std_vals,
        'total_samples': n_samples
    }
    
    return result


def main():
    print("\n" + "=" * 80)
    print("CHUNKED PROCESSING EXAMPLE (BAD VERSION)")
    print("=" * 80)
    print("\nThis version loads entire dataset into memory - memory intensive!")
    print("Watch for large memory spike (~400 MB) when creating full array")
    print("=" * 80)
    
    result = analyze_large_dataset_all_at_once(n_samples=5000000)
    print(f"\nProcessed {result['total_samples']:,} samples")
    print(f"Mean: {result['mean'][:3]}...")
    
    print("\n" + "=" * 80)
    print("WHAT HAPPENED")
    print("=" * 80)
    print("""
Large memory spike because:
- Created full 5M × 10 array = ~400 MB
- All data loaded into memory at once
- Fine for small datasets, problematic for large ones

Next, run the GOOD version to see the fix:
    python -m memory_profiler chunked_processing_good.py
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
