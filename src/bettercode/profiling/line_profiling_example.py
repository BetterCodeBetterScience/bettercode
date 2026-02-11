"""
Line Profiling Example with line_profiler

This script demonstrates line-by-line profiling using the line_profiler package.
Line profiling shows exactly which lines of code consume the most time.

Installation:
    pip install line_profiler

Usage:
    # Method 1: Using kernprof (recommended)
    kernprof -lv line_profiling_example.py
    
    # Method 2: Run directly (uses manual profiling)
    python line_profiling_example.py

The @profile decorator is recognized by kernprof but will cause an error
if you run the script directly without defining it. This script handles both cases.
"""

import numpy as np

# Handle @profile decorator for both kernprof and direct execution
try:
    # If running with kernprof, @profile is already defined
    profile
except NameError:
    # If running directly, create a no-op decorator
    def profile(func):
        return func


@profile
def find_duplicates_inefficient(data):
    duplicates = []
    seen = []
    
    for item in data:
        if item in seen: 
            if item not in duplicates:  
                duplicates.append(item)
        else:
            seen.append(item)
    
    return duplicates


@profile
def find_duplicates_efficient(data):
    duplicates = set()
    seen = set()
    
    for item in data:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    return list(duplicates)


@profile
def process_nested_data(data_size=1000):
    """Example with nested loops to show line-by-line time distribution.
    
    Line profiling will reveal which operations in nested loops
    consume the most time.
    """
    # Create 2D data
    matrix = np.random.rand(data_size, 100)
    
    # Process each row
    results = []
    for row in matrix:
        # Calculate statistics for each row
        mean_val = np.mean(row)
        std_val = np.std(row)
        max_val = np.max(row)
        
        # Combine results
        if mean_val > 0.5:
            results.append(mean_val * std_val + max_val)
        else:
            results.append(mean_val + std_val * max_val)
    
    return results


def main():
    """Run profiling examples."""
    print("=" * 80)
    print("Line Profiling Examples")
    print("=" * 80)
    print("\nThis script demonstrates line-by-line performance profiling.")
    print("For best results, run with kernprof:")
    print("  kernprof -lv line_profiling_example.py")
    print("\nThe @profile decorator marks functions for line-by-line analysis.")
    print("=" * 80)
    
    # Generate test data
    data_size = 10000
    data = list(range(data_size)) + list(range(data_size // 2))
    np.random.shuffle(data)
    
    print(f"\n1. Testing inefficient duplicate finding (data_size={len(data)})...")
    result1 = find_duplicates_inefficient(data)
    print(f"   Found {len(result1)} duplicates")
    
    print(f"\n2. Testing efficient duplicate finding (data_size={len(data)})...")
    result2 = find_duplicates_efficient(data)
    print(f"   Found {len(result2)} duplicates")
    
    print("\n3. Testing nested data processing (matrix_size=1000x100)...")
    result3 = process_nested_data(1000)
    print(f"   Processed {len(result3)} rows")
    
    print("\n" + "=" * 80)
    print("Execution complete!")
    print("=" * 80)
    
    if 'profile' not in dir(__builtins__):
        print("\nNOTE: Line-by-line profiling output not shown.")
        print("To see detailed line profiling, run:")
        print("  kernprof -lv line_profiling_example.py")
        print("\nThis will create a .lprof file and display:")
        print("  - Time spent on each line")
        print("  - Number of times each line executed")
        print("  - Percentage of total time per line")
    print("=" * 80)


if __name__ == "__main__":
    main()
