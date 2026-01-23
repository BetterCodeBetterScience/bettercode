"""
Profiling Examples with cProfile

This script demonstrates function profiling using Python's built-in cProfile module.
Run this script to see detailed profiling output without IPython overhead.

Usage:
    python profiling_example.py
"""

import cProfile
import pstats
import io
import json
import tempfile
import os
import numpy as np


def fibonacci_recursive(n):
    """Inefficient recursive implementation of fibonacci."""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n):
    """Efficient iterative implementation of fibonacci."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def matrix_operations():
    """Perform various matrix operations."""
    # Create large matrices
    size = 500
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    # Matrix multiplication
    C = np.dot(A, B)

    # Matrix inversion
    A_inv = np.linalg.inv(A)

    # Eigenvalue computation
    eigenvalues = np.linalg.eigvals(C[:100, :100])  # Use smaller matrix for speed

    return C, A_inv, eigenvalues


def data_processing():
    """Simulate data processing with loops."""
    data = []
    for i in range(10000):
        # Inefficient list operations
        data.append(i**2)

    # Process data
    result = []
    for value in data:
        if value % 2 == 0:
            result.append(np.sqrt(value))

    return result


def find_duplicates_inefficient(data):
    """Secretly expensive: repeated list membership checks.
    
    Using 'in' operator on lists is O(n), making this O(n²) overall.
    This is a common pattern that looks innocent but scales poorly.
    """

    duplicates = []
    seen = []
    
    for item in data:
        if item in seen:  # O(n) operation on a list!
            if item not in duplicates:  # Another O(n) operation!
                duplicates.append(item)
        else:
            seen.append(item)
    
    return duplicates


def find_duplicates_efficient(data):
    """Efficient version using set for O(1) lookups.
    
    Using sets for membership testing is O(1), making this O(n) overall.
    """

    duplicates = set()
    seen = set()
    
    for item in data:
        if item in seen:  # O(1) operation on a set!
            duplicates.add(item)
        else:
            seen.add(item)
    
    return list(duplicates)


def process_json_files_inefficient(num_files=100):
    """I/O bound: reading and parsing JSON files one at a time.
    
    The performance bottleneck here isn't obvious - it's the
    repeated file open/close operations and JSON parsing.
    """
    # Create temporary directory and files
    temp_dir = tempfile.mkdtemp()
    
    # Write test data
    for i in range(num_files):
        data = {"id": i, "values": list(range(100)), "metadata": {"count": 100}}
        filepath = os.path.join(temp_dir, f"data_{i}.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
    
    # Read files one by one (inefficient I/O pattern)
    results = []
    for i in range(num_files):
        filepath = os.path.join(temp_dir, f"data_{i}.json")
        with open(filepath, "r") as f:
            data = json.load(f)
            results.append(sum(data["values"]))
    
    # Cleanup
    for i in range(num_files):
        os.remove(os.path.join(temp_dir, f"data_{i}.json"))
    os.rmdir(temp_dir)
    
    return results


def process_json_files_efficient(num_files=100):
    """More efficient: minimize I/O operations and use efficient parsing.
    
    Batching operations and keeping files open longer can help,
    but the real lesson is that I/O is often the bottleneck.
    """
    # Create temporary directory and files
    temp_dir = tempfile.mkdtemp()
    
    # Write test data (same as inefficient version)
    for i in range(num_files):
        data = {"id": i, "values": list(range(100)), "metadata": {"count": 100}}
        filepath = os.path.join(temp_dir, f"data_{i}.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
    
    # Read files - same pattern but the point is to show I/O impact
    results = []
    for i in range(num_files):
        filepath = os.path.join(temp_dir, f"data_{i}.json")
        with open(filepath, "r") as f:
            data = json.load(f)
            results.append(sum(data["values"]))
    
    # Cleanup
    for i in range(num_files):
        os.remove(os.path.join(temp_dir, f"data_{i}.json"))
    os.rmdir(temp_dir)
    
    return results


def run_function_profiling():
    """Example function that calls other functions."""
    # Test fibonacci implementations
    result1 = fibonacci_recursive(20)
    result2 = fibonacci_iterative(20)

    # Run matrix operations
    matrix_results = matrix_operations()

    # Run data processing
    processed_data = data_processing()
    
    # Run JSON I/O (I/O bound)
    json_results = process_json_files_inefficient(50)

    return result1, result2, matrix_results, processed_data, duplicates, json_results


def profile_main_function():
    """Profile the main function with detailed output."""
    print("=" * 80)
    print("Profiling complete workflow")
    print("=" * 80)

    profiler = cProfile.Profile()
    profiler.enable()

    # Run the function
    results = run_function_profiling()

    profiler.disable()

    # Display results sorted by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # Show top 20 functions
    print(s.getvalue())

    return results


def compare_fibonacci_implementations():
    """Compare recursive vs iterative fibonacci implementations."""
    print("\n" + "=" * 80)
    print("Comparing Fibonacci Implementations")
    print("=" * 80)

    # Profile recursive version
    print("\nProfiling recursive fibonacci (n=25):")
    print("-" * 80)
    profiler = cProfile.Profile()
    profiler.enable()
    result = fibonacci_recursive(25)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    print(s.getvalue())
    print(f"Result: {result}")

    # Profile iterative version
    print("\nProfiling iterative fibonacci (n=25):")
    print("-" * 80)
    profiler = cProfile.Profile()
    profiler.enable()
    result = fibonacci_iterative(25)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    print(s.getvalue())
    print(f"Result: {result}")


def compare_duplicate_finding(data_size=10000):
    """Compare list vs set for membership testing - a common performance trap!"""
    print("\n" + "=" * 80)
    print("Comparing Data Structures: List vs Set (Common Performance Trap!)")
    print("=" * 80)
    print("\nUsing 'in' operator on lists is O(n), but on sets it's O(1).")
    print("This makes a huge difference when checking membership repeatedly!\n")

    # Profile inefficient version with list
    print(f"\nProfiling duplicate finding with list (data_size={data_size}):")
    print("(Using 'if item in seen_list' is O(n) each time)")
    print("-" * 80)
    data = list(range(data_size)) + list(range(data_size // 2))
    np.random.shuffle(data)
    
    profiler = cProfile.Profile()
    profiler.enable()
    result = find_duplicates_inefficient(data)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    print(s.getvalue())
    print(f"Found {len(result)} duplicates")

    # Profile efficient version with set
    print(f"\nProfiling duplicate finding with set (data_size={data_size}):")
    print("(Using 'if item in seen_set' is O(1) each time)")
    print("-" * 80)
    profiler = cProfile.Profile()
    profiler.enable()
    result = find_duplicates_efficient(data)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    print(s.getvalue())
    print(f"Found {len(result)} duplicates")
    
    print("\n" + "-" * 80)
    print("KEY LESSON: Choosing the right data structure matters enormously!")
    print("  - List membership check: O(n) - must scan entire list")
    print("  - Set membership check: O(1) - instant hash lookup")
    print("  - In a loop: O(n²) vs O(n) - dramatic difference at scale")
    print("  - This is one of the most common performance mistakes in Python!")
    print("=" * 80)


def compare_io_operations():
    """Profile I/O-bound operations - bottlenecks aren't always in your code!"""
    print("\n" + "=" * 80)
    print("Profiling I/O-Bound Operations")
    print("=" * 80)
    print("\nI/O operations are often the bottleneck, but it's not obvious")
    print("from the code. Profiling reveals time spent in file operations.\n")

    # Profile JSON file processing
    print("\nProfiling JSON file processing (50 files):")
    print("-" * 80)
    profiler = cProfile.Profile()
    profiler.enable()
    result = process_json_files_inefficient(50)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())
    print(f"Processed {len(result)} files")
    print("\nNotice the time spent in 'open', 'load', and other I/O operations!")


def profile_individual_functions():
    """Profile each function individually for detailed analysis."""
    print("\n" + "=" * 80)
    print("Profiling Individual Functions")
    print("=" * 80)

    # Profile matrix operations
    print("\nProfiling matrix_operations():")
    print("-" * 80)
    profiler = cProfile.Profile()
    profiler.enable()
    matrix_operations()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())

    # Profile data processing
    print("\nProfiling data_processing():")
    print("-" * 80)
    profiler = cProfile.Profile()
    profiler.enable()
    data_processing()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())


def main():
    """Run all profiling examples."""
    print("\n" + "=" * 80)
    print("Python Function Profiling Examples with cProfile")
    print("=" * 80)
    print("\nThis script demonstrates profiling techniques for identifying")
    print("performance bottlenecks in Python code.")
    print("=" * 80)

    # Profile the complete workflow
    # profile_main_function()

    # Compare fibonacci implementations
    # compare_fibonacci_implementations()
    
    # Compare data structures (list vs set - subtle performance trap)
    compare_duplicate_finding()
    
    # Compare I/O operations (I/O bound example)
    compare_io_operations()

    # Profile individual functions
    profile_individual_functions()

    print("\n" + "=" * 80)
    print("Profiling Complete")
    print("=" * 80)
    print("\nKey metrics in the output:")
    print("  ncalls  - Number of calls")
    print("  tottime - Total time in function (excluding subcalls)")
    print("  percall - tottime / ncalls")
    print("  cumtime - Cumulative time (including subcalls)")
    print("  percall - cumtime / ncalls")
    print("\nKey lessons:")
    print("  1. Obvious slowness (fibonacci): Recursive calls create huge overhead")
    print("  2. Subtle slowness (list vs set): Wrong data structure = O(n²) instead of O(n)")
    print("  3. I/O bound (JSON files): File operations dominate execution time")
    print("  4. Always profile - don't assume you know where the bottleneck is!")
    print("=" * 80)


if __name__ == "__main__":
    main()
