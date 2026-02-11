"""
Memory Profiling Example 4b: Array Growth (GOOD VERSION)

This script demonstrates preallocating arrays when the size is known.

Usage:
    python -m memory_profiler array_growth_good.py

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
def preallocate_array(n_iterations: int = 1000) -> np.ndarray:
    """Efficiently build array by preallocating memory.
    
    Parameters
    ----------
    n_iterations : int, default=1000
        Number of batches to generate
    
    Returns
    -------
    np.ndarray
        Final array with all batches
    """
    batch_size = 1000
    final_array = np.empty(n_iterations * batch_size)
    
    for i in range(n_iterations):
        start = i * batch_size
        end = start + batch_size
        final_array[start:end] = np.random.randn(batch_size)
    
    return final_array


def main() -> None:
    """Run the good array growth example and explain the benefits."""
    print("\n" + "=" * 80)
    print("ARRAY GROWTH EXAMPLE (GOOD VERSION)")
    print("=" * 80)
    print("\nThis version preallocates the array!")
    print("Watch memory allocated once and filled in-place")
    print("=" * 80)
    
    result = preallocate_array(n_iterations=1000)
    print(f"\nCreated array of size {len(result):,}")
    
    print("\n" + "=" * 80)
    print("WHAT HAPPENED")
    print("=" * 80)
    print("""
Memory efficient because:
- final_array = np.empty(total_size) allocates once
- Loop fills array in-place with final_array[start:end] = batch
- No concatenation, no copying, no memory spikes
- Faster execution too!

KEY INSIGHT:
- BAD: results.append(batch); np.concatenate(results)
- GOOD: results = np.empty(total_size); results[i] = batch

This pattern matters when:
- Building arrays/lists in a loop
- You know the final size in advance
- Appending/concatenating large arrays

Why append is inefficient:
- Lists grow by allocating larger memory, copying all data
- np.concatenate creates new array, copies all input arrays
- With 1000 iterations, data is copied many times

Preallocation benefits:
- Memory allocated once
- Data written directly to final location
- No copying overhead
- Faster execution and lower peak memory

When preallocation isn't possible:
- Use list.append() for small Python objects
- Use deque for efficient append/pop on both ends
- Consider numpy.vstack() with smaller batches
- For truly unknown sizes, grow in chunks (e.g., double size)

Performance impact: Can be 10-100x faster for large arrays!
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
