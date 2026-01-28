"""
Memory Profiling Example 2b: Chunked Processing (GOOD VERSION)

This script demonstrates processing a large dataset in chunks.

Usage:
    python -m memory_profiler chunked_processing_good.py

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
def analyze_large_dataset_chunked(n_samples=5000000, chunk_size=500000):
    n_chunks = n_samples // chunk_size
    
    n_seen = 0
    sum_vals = None
    sum_sq_vals = None
    
    for i in range(n_chunks):
        chunk = np.random.randn(chunk_size, 10)
        
        if sum_vals is None:
            sum_vals = np.sum(chunk, axis=0)
            sum_sq_vals = np.sum(chunk ** 2, axis=0)
        else:
            sum_vals += np.sum(chunk, axis=0)
            sum_sq_vals += np.sum(chunk ** 2, axis=0)
        
        n_seen += chunk_size
    
    # Calculate final statistics
    mean_vals = sum_vals / n_seen
    std_vals = np.sqrt(sum_sq_vals / n_seen - mean_vals ** 2)
    
    result = {
        'mean': mean_vals,
        'std': std_vals,
        'total_samples': n_seen
    }
    
    return result


def main():
    print("\n" + "=" * 80)
    print("CHUNKED PROCESSING EXAMPLE (GOOD VERSION)")
    print("=" * 80)
    print("\nThis version processes data in chunks - memory efficient!")
    print("Watch memory stay low (~40 MB per chunk) throughout processing")
    print("=" * 80)
    
    result = analyze_large_dataset_chunked(n_samples=5000000, chunk_size=500000)
    print(f"\nProcessed {result['total_samples']:,} samples")
    print(f"Mean: {result['mean'][:3]}...")
    
    print("\n" + "=" * 80)
    print("WHAT HAPPENED")
    print("=" * 80)
    print("""
Memory stayed low because:
- Processed in 10 chunks of 500K samples each
- Only one chunk (~40 MB) in memory at a time
- Used online algorithm for mean/std calculation
- 10x less memory than loading all at once!

KEY INSIGHT:
- BAD: data = load_all_data() - requires RAM for entire dataset
- GOOD: for chunk in chunks: process(chunk) - constant memory

This pattern is essential when:
- Working with datasets larger than available RAM
- Processing CSV files, databases, or streaming data
- You can compute statistics incrementally (online algorithms)

Many operations support online/incremental computation:
- Mean, variance, standard deviation
- Min, max, sum, count
- Histograms with fixed bins
- Many machine learning algorithms
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
