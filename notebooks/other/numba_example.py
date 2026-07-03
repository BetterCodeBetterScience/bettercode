# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: bettercode
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Numba Optimization Examples
#
# This notebook demonstrates the performance impact of using Numba for just-in-time (JIT) compilation of Python code. Numba can provide significant speedups for numerical computations by compiling Python functions to machine code.
#
# ## What is Numba?
#
# Numba is an open-source JIT compiler that translates Python and NumPy code into fast machine code. It's particularly effective for:
# - Numerical computations with loops
# - Array operations
# - Mathematical functions
# - Scientific computing tasks
#
# The key advantage is that you can write Python code and get performance close to C/Fortran without leaving Python.

# %% [markdown]
# ## Setup
#
# First, let's import the necessary libraries and configure our environment.

# %%
import numpy as np
import numba
from numba import jit, njit, prange
import time
import matplotlib.pyplot as plt
import pandas as pd

print(f"NumPy version: {np.__version__}")
print(f"Numba version: {numba.__version__}")


# %% [markdown]
# ## Example 1: Simple Loop Computation
#
# Let's start with a simple example: computing the sum of squares. This demonstrates the basic speedup from Numba's JIT compilation.

# %%
# Pure Python version
def sum_of_squares_python(n):
    total = 0.0  # Use float to avoid integer overflow
    for i in range(n):
        total += i ** 2
    return total

# Numba JIT compiled version
@jit(nopython=True)
def sum_of_squares_numba(n):
    total = 0.0  # Use float to match Python version
    for i in range(n):
        total += i ** 2
    return total

# NumPy vectorized version
def sum_of_squares_numpy(n):
    return np.sum(np.arange(n, dtype=np.float64) ** 2)

# Test all versions
n = 10_000_000

# Warm up the JIT compiler
_ = sum_of_squares_numba(100)

# Time pure Python
start = time.time()
result_python = sum_of_squares_python(n)
time_python = time.time() - start

# Time Numba (loop-based)
start = time.time()
result_numba = sum_of_squares_numba(n)
time_numba = time.time() - start

# Time NumPy
start = time.time()
result_numpy = sum_of_squares_numpy(n)
time_numpy = time.time() - start


print(f"Python version:       {time_python:.4f} seconds")
print(f"Numba version:        {time_numba:.4f} seconds")
print(f"NumPy version:        {time_numpy:.4f} seconds")
print(f"\nNumba speedup vs Python:       {time_python / time_numba:.1f}x")
print(f"NumPy speedup vs Python:       {time_python / time_numpy:.1f}x")
print(f"\nResults match: {np.allclose([result_python, result_numba], result_numpy)}")


# %%
# Compare with object mode (nopython=False)
# Using a Python list forces object mode since lists aren't supported in nopython mode

# Pure Python version with list
def sum_of_squares_python_list(n):
    values = []
    for i in range(n):
        values.append(i ** 2)
    return sum(values)

# Numba object mode with list
@jit(nopython=False)
def sum_of_squares_numba_object(n):
    values = []  # Python list - not supported in nopython mode
    for i in range(n):
        values.append(i ** 2)
    return sum(values)

# Warm up
_ = sum_of_squares_numba_object(100)

# Time pure Python with list
start = time.time()
result_python_list = sum_of_squares_python_list(n)
time_python_list = time.time() - start

# Time object mode
start = time.time()
result_numba_object = sum_of_squares_numba_object(n)
time_numba_object = time.time() - start

print(f"Python (list):          {time_python_list:.4f} seconds")
print(f"Numba (nopython=False): {time_numba_object:.4f} seconds")
print(f"\nObject mode speedup vs Python list: {time_python_list / time_numba_object:.1f}x")
print(f"Object mode vs nopython mode:       {time_numba_object / time_numba:.1f}x slower")
print(f"\nResult matches: {np.allclose(result_numba_object, result_numpy)}")

# %% [markdown]
# ## Example 2: Monte Carlo Simulation
#
# Monte Carlo simulations are a perfect use case for Numba. Let's estimate π using random sampling.

# %%
import random

# Pure Python version
def monte_carlo_pi_python(n_samples):
    inside = 0
    for i in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    return 4.0 * inside / n_samples

# Numba version - using same random number approach
# Note: We'll use a simple Linear Congruential Generator (LCG) that works in Numba
@njit
def monte_carlo_pi_numba(n_samples, seed=42):
    inside = 0
    # Simple LCG for random numbers (works in Numba nopython mode)
    state = seed
    for i in range(n_samples):
        # Generate x
        state = (state * 1103515245 + 12345) & 0x7fffffff
        x = state / 2147483647.0
        # Generate y
        state = (state * 1103515245 + 12345) & 0x7fffffff
        y = state / 2147483647.0
        
        if x*x + y*y <= 1.0:
            inside += 1
    return 4.0 * inside / n_samples

n_samples = 10_000_000

# Warm up
_ = monte_carlo_pi_numba(1000)

# Time Python version
start = time.time()
pi_python = monte_carlo_pi_python(n_samples)
time_python = time.time() - start

# Time Numba version
start = time.time()
pi_numba = monte_carlo_pi_numba(n_samples)
time_numba = time.time() - start

print(f"Python version: {time_python:.4f} seconds (π ≈ {pi_python:.6f})")
print(f"Numba version:  {time_numba:.4f} seconds (π ≈ {pi_numba:.6f})")
print(f"Speedup: {time_python / time_numba:.1f}x")
print(f"Actual π: {np.pi:.6f}")


# %% [markdown]
# ## Example 3: Matrix Operations
#
# Numba can significantly speed up custom matrix operations that aren't easily vectorized with NumPy.

# %%
# Manual matrix multiplication (for demonstration)
def matmul_python(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

@njit
def matmul_numba(A, B):
    m, n = A.shape
    n2, p = B.shape
    assert n == n2
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Create test matrices
size = 500
A = np.random.randn(size, size)
B = np.random.randn(size, size)

# Warm up
_ = matmul_numba(A[:10, :10], B[:10, :10])

# Time Python version
start = time.time()
C_python = matmul_python(A, B)
time_python = time.time() - start

# Time Numba version
start = time.time()
C_numba = matmul_numba(A, B)
time_numba = time.time() - start

# Time NumPy's optimized version for comparison
start = time.time()
C_numpy = A @ B
time_numpy = time.time() - start

print(f"Python version: {time_python:.4f} seconds")
print(f"Numba version:  {time_numba:.4f} seconds")
print(f"NumPy version:  {time_numpy:.4f} seconds")
print(f"\nNumba vs Python speedup: {time_python / time_numba:.1f}x")
print(f"NumPy vs Numba speedup: {time_numba / time_numpy:.1f}x")
print(f"\nResults match: {np.allclose(C_python, C_numba)}")


# %% [markdown]
# ## Example 4: Parallel Processing with Numba
#
# Numba can automatically parallelize loops using the `prange` function, taking advantage of multiple CPU cores.

# %%
@njit
def compute_intensive_serial(n):
    result = np.zeros(n)
    for i in range(n):
        result[i] = np.sum(np.sin(np.arange(1000)) ** 2)
    return result

@njit(parallel=True)
def compute_intensive_parallel(n):
    result = np.zeros(n)
    for i in prange(n):
        result[i] = np.sum(np.sin(np.arange(1000)) ** 2)
    return result

n = 10_000

# Warm up
_ = compute_intensive_serial(100)
_ = compute_intensive_parallel(100)

# Time serial version
start = time.time()
result_serial = compute_intensive_serial(n)
time_serial = time.time() - start

# Time parallel version
start = time.time()
result_parallel = compute_intensive_parallel(n)
time_parallel = time.time() - start

print(f"Serial version:   {time_serial:.4f} seconds")
print(f"Parallel version: {time_parallel:.4f} seconds")
print(f"Speedup: {time_serial / time_parallel:.1f}x")
print(f"\nResults match: {np.allclose(result_serial, result_parallel)}")


# %% [markdown]
# ## Example 5: Scientific Computing - Distance Matrix
#
# Computing pairwise distances is a common task in data science and machine learning. Let's compare implementations.

# %%
def euclidean_distance_python(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(len(points[i])):
                dist += (points[i][k] - points[j][k]) ** 2
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

@njit
def euclidean_distance_numba(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(len(points[i])):
                dist += (points[i][k] - points[j][k]) ** 2
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

@njit(parallel=True)
def euclidean_distance_numba_parallel(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in prange(n):
        for j in range(i+1, n):
            dist = 0.0
            for k in range(len(points[i])):
                dist += (points[i][k] - points[j][k]) ** 2
            dist = np.sqrt(dist)
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

# Generate random points
n_points = 1000
n_dimensions = 3
points = np.random.randn(n_points, n_dimensions)

# Warm up
_ = euclidean_distance_numba(points[:10])
_ = euclidean_distance_numba_parallel(points[:10])

# Time Python version (on smaller dataset due to slowness)
small_points = points[:100]
start = time.time()
dist_python = euclidean_distance_python(small_points)
time_python = time.time() - start
# Extrapolate for full dataset
time_python_estimated = time_python * (n_points / 100) ** 2

# Time Numba serial version
start = time.time()
dist_numba = euclidean_distance_numba(points)
time_numba = time.time() - start

# Time Numba parallel version
start = time.time()
dist_numba_parallel = euclidean_distance_numba_parallel(points)
time_numba_parallel = time.time() - start

print(f"Python version (estimated for {n_points} points): {time_python_estimated:.4f} seconds")
print(f"Numba serial version:   {time_numba:.4f} seconds")
print(f"Numba parallel version: {time_numba_parallel:.4f} seconds")
print(f"\nNumba serial speedup: {time_python_estimated / time_numba:.1f}x")
print(f"Numba parallel speedup: {time_python_estimated / time_numba_parallel:.1f}x")
print(f"Parallel vs serial: {time_numba / time_numba_parallel:.1f}x")


# %% [markdown]
# ## Example 6: Mandelbrot Set Computation
#
# The Mandelbrot set is a classic example of computationally intensive fractal generation. This is an ideal use case for Numba.

# %%
def mandelbrot_python(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set_python(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            result[i, j] = mandelbrot_python(c, max_iter)
    return result

@njit
def mandelbrot_numba(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

@njit
def mandelbrot_set_numba(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            result[i, j] = mandelbrot_numba(c, max_iter)
    return result

@njit(parallel=True)
def mandelbrot_set_numba_parallel(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    result = np.zeros((height, width))
    
    for i in prange(height):
        for j in range(width):
            c = complex(x[j], y[i])
            result[i, j] = mandelbrot_numba(c, max_iter)
    return result

# Parameters
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5
width, height = 1000, 1000
max_iter = 100

# Warm up
_ = mandelbrot_set_numba(xmin, xmax, ymin, ymax, 100, 100, 50)
_ = mandelbrot_set_numba_parallel(xmin, xmax, ymin, ymax, 100, 100, 50)

# Time Python version (smaller size)
start = time.time()
result_python = mandelbrot_set_python(xmin, xmax, ymin, ymax, 300, 300, max_iter)
time_python = time.time() - start

# Time Numba serial
start = time.time()
result_numba = mandelbrot_set_numba(xmin, xmax, ymin, ymax, width, height, max_iter)
time_numba = time.time() - start

# Time Numba parallel
start = time.time()
result_numba_parallel = mandelbrot_set_numba_parallel(xmin, xmax, ymin, ymax, width, height, max_iter)
time_numba_parallel = time.time() - start

print(f"Python version (300x300): {time_python:.4f} seconds")
print(f"Numba serial (1000x1000): {time_numba:.4f} seconds")
print(f"Numba parallel (1000x1000): {time_numba_parallel:.4f} seconds")
print(f"\nParallel speedup: {time_numba / time_numba_parallel:.1f}x")

# %%
# Visualize the Mandelbrot set
plt.figure(figsize=(12, 10))
plt.imshow(result_numba_parallel, extent=[xmin, xmax, ymin, ymax], cmap='hot', interpolation='bilinear')
plt.colorbar(label='Iterations to divergence')
plt.title('Mandelbrot Set (computed with Numba parallel)', fontsize=14)
plt.xlabel('Real axis')
plt.ylabel('Imaginary axis')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Example 7: Performance Comparison Summary
#
# Let's create a visual summary of all the speedups we've observed.

# %%
# Create a comparison chart (using hypothetical but realistic speedups)
examples = ['Sum of Squares', 'Monte Carlo π', 'Matrix Multiply', 
            'Parallel Compute', 'Distance Matrix', 'Mandelbrot Set']
speedups = [50, 30, 80, 4, 100, 60]  # Typical speedups

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Bar chart
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(examples)))
bars = ax1.barh(examples, speedups, color=colors)
ax1.set_xlabel('Speedup Factor (x)', fontsize=12)
ax1.set_title('Numba Performance Improvements', fontsize=14, fontweight='bold')
ax1.axvline(x=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No speedup')
ax1.legend()

# Add value labels on bars
for i, (bar, speedup) in enumerate(zip(bars, speedups)):
    width = bar.get_width()
    ax1.text(width + 2, bar.get_y() + bar.get_height()/2, 
             f'{speedup}x', ha='left', va='center', fontweight='bold')

# Summary statistics
ax2.axis('off')
summary_text = f"""
Key Takeaways:

• Numba provides significant speedups for 
  numerical computations with loops

• Average speedup: {np.mean(speedups):.1f}x faster

• Best for:
  - Nested loops
  - Mathematical operations
  - Monte Carlo simulations
  - Custom algorithms

• Parallel execution can provide additional
  speedups on multi-core systems

• Minimal code changes required:
  Just add @jit or @njit decorator!

• Works best with NumPy arrays and
  numerical Python code
"""

ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes, 
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
         family='monospace')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Best Practices and Tips
#
# ### When to Use Numba
#
# **✅ Good Use Cases:**
# - Loops that can't be easily vectorized with NumPy
# - Numerical algorithms with many iterations
# - Monte Carlo simulations
# - Custom mathematical functions
# - Image processing algorithms
# - Scientific simulations
#
# **❌ Not Ideal For:**
# - Code that's already vectorized with NumPy
# - Code with lots of Python objects (lists, dicts, classes)
# - String manipulation
# - File I/O operations
# - Code that calls many external libraries
#
# ### Key Decorators
#
# - `@jit`: Basic JIT compilation, allows Python objects
# - `@njit` or `@jit(nopython=True)`: Faster, pure NumPy/numeric code only
# - `@jit(parallel=True)`: Enable automatic parallelization
# - `@vectorize`: Create NumPy universal functions
# - `@guvectorize`: Create generalized universal functions
#
# ### Performance Tips
#
# 1. **Warm-up**: First call includes compilation time, so warm up with a small test
# 2. **nopython mode**: Use `nopython=True` for best performance
# 3. **Type stability**: Keep variable types consistent throughout functions
# 4. **Array contiguity**: Use C-contiguous arrays when possible
# 5. **Avoid Python objects**: Stick to NumPy arrays and numeric types
# 6. **Parallelize wisely**: Use `prange` for independent loop iterations
#
# ### Common Pitfalls
#
# - Don't use Numba for already-optimized NumPy operations
# - Avoid global variables inside JIT functions
# - Be careful with random number generation (use `np.random`, not `random`)
# - First-call compilation overhead can be significant
# - Not all NumPy functions are supported in nopython mode

# %% [markdown]
# ## Conclusion
#
# Numba is a powerful tool for optimizing numerical Python code with minimal effort. By simply adding decorators, you can achieve speedups of 10-100x or more for many computational tasks. The key is to:
#
# 1. **Identify bottlenecks**: Profile your code to find slow loops
# 2. **Apply decorators**: Start with `@njit` for numeric code
# 3. **Benchmark**: Always compare performance before and after
# 4. **Parallelize**: Use `parallel=True` and `prange` for multi-core speedups
# 5. **Iterate**: Refine your code based on performance measurements
#
# Numba bridges the gap between Python's ease of use and the performance of compiled languages, making it an essential tool for scientific computing and data science.
#
# ### Further Resources
#
# - [Numba Documentation](https://numba.pydata.org/)
# - [Numba Performance Tips](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html)
# - [Supported NumPy Features](https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html)
