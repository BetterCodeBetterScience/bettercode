"""
Memory Profiling Example 3b: Unnecessary Copies (GOOD VERSION)

This script demonstrates using in-place operations to avoid unnecessary copies.

Usage:
    python -m memory_profiler unnecessary_copies_good.py

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
def process_without_copies(n_rows=2000000):
    data = np.random.randn(n_rows, 5) 
    
    mean = data.mean()
    std = data.std()
    data_normalized = (data - mean) / std  
    data_normalized **= 2  
    
    result = data_normalized.sum(axis=1)
    
    return result


def main():
    print("\n" + "=" * 80)
    print("UNNECESSARY COPIES EXAMPLE (GOOD VERSION)")
    print("=" * 80)
    print("\nThis version uses in-place operations to avoid copies!")
    print("Watch memory peak at ~160 MB (only 2 arrays needed)")
    print("=" * 80)
    
    result = process_without_copies(n_rows=2000000)
    print(f"\nProcessed {len(result):,} rows")
    
    print("\n" + "=" * 80)
    print("WHAT HAPPENED")
    print("=" * 80)
    print("""
Memory stayed at ~160 MB because:
- data = np.random.randn(...) = 80 MB
- data_normalized = (data - mean) / std = another 80 MB (needed)
- data_normalized **= 2 (in-place, no copy!)
- No unnecessary copies
- Peak memory: 2 × 80 MB = ~160 MB

KEY INSIGHT:
- BAD: data_copy = data.copy() at every step
- GOOD: data **= 2 (in-place) or work with original when safe

This pattern matters when:
- Working with large arrays/DataFrames
- Performing multiple transformations
- "Defensive copying" becomes excessive

When to copy vs. not copy:
- Copy when you need to preserve the original data
- Don't copy when you're done with the original
- Use in-place operations (+=, -=, **=, etc.) when possible
- Understand numpy views vs. copies

Memory savings: 2x or more (320 MB → 160 MB in this example)

Tips:
- numpy arrays: Use arr **= 2 instead of arr = arr ** 2
- pandas DataFrames: Use inplace=True when available
- Check if operations return views or copies (numpy docs)
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
