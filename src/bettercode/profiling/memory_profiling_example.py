"""
Memory Profiling: Finding Memory Leaks and Inefficiencies

This script demonstrates realistic scenarios where line-by-line memory profiling
with memory_profiler is essential for finding memory issues:

1. Memory leak: Accumulating data in a cache without cleanup
2. Inefficient data loading: Loading entire dataset vs. chunked processing
3. Unnecessary copies: Creating redundant data copies

These are common issues in scientific computing that memory profiling can reveal.

Usage:
    python -m memory_profiler memory_profiling_example.py

Install: pip install memory_profiler
"""

import numpy as np
import time

try:
    from memory_profiler import profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    def profile(func):
        return func


# ==============================================================================
# Example 1: Memory Leak - Accumulating Results Without Cleanup
# ==============================================================================

@profile
def process_data_with_leak(n_iterations=10):
    """Process data while accumulating results - memory leak!
    
    This simulates a common mistake: storing all intermediate results
    when you only need the final aggregate.
    """
    # This list will grow indefinitely - memory leak!
    all_results = []
    
    for i in range(n_iterations):
        # Simulate processing large arrays
        data = np.random.randn(1000000)  # 1M floats = ~8 MB
        result = data ** 2
        
        # BAD: Store everything (memory leak)
        all_results.append(result)
        
        # Simulate some processing time
        time.sleep(0.1)
    
    # We only needed the final sum, not all intermediate results!
    final_sum = sum(np.sum(r) for r in all_results)
    return final_sum


@profile
def process_data_efficiently(n_iterations=10):
    """Process data efficiently - accumulate only what's needed.
    
    This is the corrected version: only keep the aggregate result.
    """
    # Only track the running sum - constant memory!
    running_sum = 0.0
    
    for i in range(n_iterations):
        # Simulate processing large arrays
        data = np.random.randn(1000000)  # 1M floats = ~8 MB
        result = data ** 2
        
        # GOOD: Accumulate only what we need
        running_sum += np.sum(result)
        
        # result goes out of scope and gets garbage collected
        time.sleep(0.1)
    
    return running_sum


# ==============================================================================
# Example 2: Loading Entire Dataset vs. Chunked Processing
# ==============================================================================

@profile
def analyze_large_dataset_all_at_once(n_samples=5000000):
    """Load and process entire dataset at once - memory intensive!
    
    This simulates loading a large dataset into memory all at once.
    """
    # Load ALL data into memory at once (BAD for large datasets)
    print(f"Loading {n_samples:,} samples into memory...")
    data = np.random.randn(n_samples, 10)  # ~400 MB
    
    # Process all at once
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    
    # Calculate some statistics
    result = {
        'mean': mean_vals,
        'std': std_vals,
        'total_samples': n_samples
    }
    
    return result


@profile
def analyze_large_dataset_chunked(n_samples=5000000, chunk_size=500000):
    """Process dataset in chunks - memory efficient!
    
    This simulates chunked processing for large datasets.
    """
    n_chunks = n_samples // chunk_size
    print(f"Processing {n_samples:,} samples in {n_chunks} chunks...")
    
    # Accumulators for online statistics
    n_seen = 0
    sum_vals = None
    sum_sq_vals = None
    
    # Process in chunks - only one chunk in memory at a time
    for i in range(n_chunks):
        # Load only one chunk at a time (~40 MB instead of 400 MB)
        chunk = np.random.randn(chunk_size, 10)
        
        # Update running statistics
        if sum_vals is None:
            sum_vals = np.sum(chunk, axis=0)
            sum_sq_vals = np.sum(chunk ** 2, axis=0)
        else:
            sum_vals += np.sum(chunk, axis=0)
            sum_sq_vals += np.sum(chunk ** 2, axis=0)
        
        n_seen += chunk_size
        # chunk goes out of scope and gets garbage collected
    
    # Calculate final statistics
    mean_vals = sum_vals / n_seen
    std_vals = np.sqrt(sum_sq_vals / n_seen - mean_vals ** 2)
    
    result = {
        'mean': mean_vals,
        'std': std_vals,
        'total_samples': n_seen
    }
    
    return result


# ==============================================================================
# Example 3: Unnecessary Data Copies
# ==============================================================================

@profile
def process_with_copies(n_rows=2000000):
    """Make unnecessary copies of data - wastes memory!
    
    This simulates a common mistake: making defensive copies when not needed.
    """
    # Create initial data
    data = np.random.randn(n_rows, 5)  # ~80 MB
    
    # BAD: Make unnecessary copies at each step
    data_copy1 = data.copy()  # Another ~80 MB
    data_normalized = (data_copy1 - data_copy1.mean()) / data_copy1.std()
    
    data_copy2 = data_normalized.copy()  # Another ~80 MB
    data_squared = data_copy2 ** 2
    
    data_copy3 = data_squared.copy()  # Another ~80 MB
    result = data_copy3.sum(axis=1)
    
    # Peak memory: ~320 MB for operations that should use ~80 MB
    return result


@profile
def process_without_copies(n_rows=2000000):
    """Work with views and in-place operations - memory efficient!
    
    This is the corrected version: avoid unnecessary copies.
    """
    # Create initial data
    data = np.random.randn(n_rows, 5)  # ~80 MB
    
    # GOOD: Work in-place or with views when possible
    mean = data.mean()
    std = data.std()
    data_normalized = (data - mean) / std  # This creates a new array, but we need it
    
    # In-place operation when possible
    data_normalized **= 2  # In-place squaring, no copy needed
    
    result = data_normalized.sum(axis=1)
    
    # Peak memory: ~160 MB instead of ~320 MB
    return result


# ==============================================================================
# Example 4: List Growing Problem
# ==============================================================================

@profile
def concatenate_with_append(n_iterations=1000):
    """Build large array by appending - inefficient memory usage!
    
    This simulates growing a list/array incrementally.
    """
    # Start with empty list
    results = []
    
    # Append many times - causes repeated reallocations
    for i in range(n_iterations):
        # Each iteration appends a 1000-element array
        batch = np.random.randn(1000)
        results.append(batch)
    
    # Convert to array at the end - another copy!
    final_array = np.concatenate(results)
    
    return final_array


@profile
def preallocate_array(n_iterations=1000):
    """Preallocate array - much more memory efficient!
    
    This is the corrected version: preallocate when size is known.
    """
    # Preallocate the full array
    batch_size = 1000
    final_array = np.empty(n_iterations * batch_size)
    
    # Fill in-place - no reallocations needed
    for i in range(n_iterations):
        start = i * batch_size
        end = start + batch_size
        final_array[start:end] = np.random.randn(batch_size)
    
    return final_array


# ==============================================================================
# Main Demonstration
# ==============================================================================

def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    """Run all memory profiling demonstrations."""
    
    if not MEMORY_PROFILER_AVAILABLE:
        print("ERROR: memory_profiler not installed!")
        print("Please install it with: pip install memory_profiler")
        print("Then run: python -m memory_profiler memory_profiling_example.py")
        return
    
    print("\n" + "=" * 80)
    print("MEMORY PROFILING: FINDING REAL MEMORY ISSUES")
    print("=" * 80)
    print("\nThis demonstrates scenarios where memory profiling reveals actual problems")
    print("Run with: python -m memory_profiler memory_profiling_example.py\n")
    
    # Example 1: Memory Leak
    print_header("EXAMPLE 1: Memory Leak - Accumulating Unnecessary Data")
    print("\nBad version (watch memory grow continuously):")
    result1 = process_data_with_leak(n_iterations=10)
    print(f"Result: {result1:.2f}")
    
    print("\nGood version (constant memory usage):")
    result2 = process_data_efficiently(n_iterations=10)
    print(f"Result: {result2:.2f}")
    
    # Example 2: Chunked Processing
    print_header("EXAMPLE 2: All-at-Once vs. Chunked Processing")
    print("\nBad version (loads everything into memory):")
    result3 = analyze_large_dataset_all_at_once(n_samples=5000000)
    print(f"Processed {result3['total_samples']:,} samples")
    
    print("\nGood version (processes in chunks):")
    result4 = analyze_large_dataset_chunked(n_samples=5000000, chunk_size=500000)
    print(f"Processed {result4['total_samples']:,} samples")
    
    # Example 3: Unnecessary Copies
    print_header("EXAMPLE 3: Unnecessary Copies vs. In-Place Operations")
    print("\nBad version (makes many copies):")
    result5 = process_with_copies(n_rows=2000000)
    print(f"Processed {len(result5):,} rows")
    
    print("\nGood version (avoids unnecessary copies):")
    result6 = process_without_copies(n_rows=2000000)
    print(f"Processed {len(result6):,} rows")
    
    # Example 4: List Growth
    print_header("EXAMPLE 4: Incremental Growth vs. Preallocation")
    print("\nBad version (grows list incrementally):")
    result7 = concatenate_with_append(n_iterations=1000)
    print(f"Created array of size {len(result7):,}")
    
    print("\nGood version (preallocates array):")
    result8 = preallocate_array(n_iterations=1000)
    print(f"Created array of size {len(result8):,}")
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. MEMORY LEAKS: Only accumulate what you actually need
   - Bad: Storing all intermediate results
   - Good: Keep running aggregates (sum, mean, etc.)

2. LARGE DATASETS: Process in chunks when possible
   - Bad: Loading entire dataset into memory
   - Good: Chunked processing with online statistics

3. UNNECESSARY COPIES: Avoid defensive copying
   - Bad: Making copies "just in case"
   - Good: Use in-place operations and views

4. ARRAY GROWTH: Preallocate when size is known
   - Bad: Growing lists with append/concatenate
   - Good: Preallocate array and fill in-place

Memory profiling with memory_profiler reveals these issues by showing:
- Where memory spikes occur
- Which lines allocate the most memory
- When memory is released (or not released)

These patterns are common in scientific computing and can be the difference
between code that runs and code that crashes with "Out of Memory" errors!
""")
    
    print("=" * 80)
    print("PROFILING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
