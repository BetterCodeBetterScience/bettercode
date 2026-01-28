"""
Base functions for parallel processing examples.

This module contains the core Mandelbrot computation functions used by
parallel processing simulations.
"""

import multiprocessing as mp
import time


def get_mandelbrot_pixel(c_real, c_imag, max_iter):
    """Calculates the escape time of a single point."""
    z_real = 0
    z_imag = 0
    for i in range(max_iter):
        # z = z^2 + c
        new_real = z_real*z_real - z_imag*z_imag + c_real
        new_imag = 2 * z_real * z_imag + c_imag
        z_real, z_imag = new_real, new_imag
        if z_real*z_real + z_imag*z_imag > 4:
            return i
    return max_iter


def compute_mandelbrot_rows(args):
    """
    Compute Mandelbrot set values for a range of rows.

    Args:
        args: Tuple of (start_row, end_row, width, height, max_iter)

    Returns:
        Sum of all escape times for these rows
    """
    start_row, end_row, width, height, max_iter = args

    # Define the complex plane region
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.0, 1.0

    total = 0
    for row in range(start_row, end_row):
        for col in range(width):
            # Map pixel to complex plane
            c_real = x_min + (x_max - x_min) * col / width
            c_imag = y_min + (y_max - y_min) * row / height

            # Compute escape time
            escape_time = get_mandelbrot_pixel(c_real, c_imag, max_iter)
            total += escape_time

    return total


def run_serial(grid_size):
    """Run computation serially (all rows processed sequentially)."""
    width, height, max_iter = grid_size

    start_time = time.time()
    # Process all rows as a single task
    result = compute_mandelbrot_rows((0, height, width, height, max_iter))
    elapsed_time = time.time() - start_time

    return result, elapsed_time


def run_parallel(grid_size, ncores=None, chunking_factor=1):
    """
    Run computation in parallel by splitting rows across workers.

    Each worker processes a chunk of rows from the grid.

    Args:
        grid_size: Tuple of (width, height, max_iter)
        ncores: Number of cores to use (default: cpu_count - 1)
        chunking_factor: Multiplier for number of chunks (n_chunks = chunking_factor * ncores)

    Returns:
        Tuple of (result, elapsed_time, n_cores)
    """
    width, height, max_iter = grid_size

    if ncores is not None:
        n_cores = ncores
    else:
        n_cores = max(1, mp.cpu_count() - 1)

    # Split rows into chunks for parallel processing
    n_chunks = chunking_factor * n_cores
    rows_per_chunk = height // n_chunks
    chunks = []
    for i in range(n_chunks):
        start_row = i * rows_per_chunk
        # Last chunk gets any remaining rows
        end_row = height if i == n_chunks - 1 else (i + 1) * rows_per_chunk
        chunks.append((start_row, end_row, width, height, max_iter))

    start_time = time.time()
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(compute_mandelbrot_rows, chunks)
    elapsed_time = time.time() - start_time

    # Sum results from all chunks
    total_result = sum(results)

    return total_result, elapsed_time, n_cores
