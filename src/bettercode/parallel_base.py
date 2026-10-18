"""
Base functions for parallel processing examples.

This module contains the core Mandelbrot computation functions used by
parallel processing simulations.
"""

import multiprocessing as mp
import time

# moving docstring out of function for direct read into latex
# """Calculate the escape time of a single point in the Mandelbrot set.

# Parameters
# ----------
# c_real : float
#     Real component of complex number c
# c_imag : float
#     Imaginary component of complex number c
# max_iter : int
#     Maximum number of iterations to test

# Returns
# -------
# int
#     Number of iterations until escape (or max_iter if no escape)
# """


def get_mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    z_real = 0
    z_imag = 0
    for i in range(max_iter):
        # z = z^2 + c
        new_real = z_real * z_real - z_imag * z_imag + c_real
        new_imag = 2 * z_real * z_imag + c_imag
        z_real, z_imag = new_real, new_imag
        if z_real * z_real + z_imag * z_imag > 4:
            return i
    return max_iter


# moving docstring out of function for direct read into latex
# """Compute Mandelbrot set values for a range of rows.

# Parameters
# ----------
# args : tuple
#     Tuple of (start_row, end_row, width, height, max_iter)

# Returns
# -------
# int
#     Sum of all escape times for these rows
# """


def compute_mandelbrot_rows(args: tuple) -> int:
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


# moving docstring out of function for direct read into latex
# """Run computation serially (all rows processed sequentially).

# Parameters
# ----------
# grid_size : tuple[int, int, int]
#     Tuple of (width, height, max_iter)

# Returns
# -------
# tuple[int, float]
#     Tuple of (result sum, elapsed time in seconds)
# """


def run_serial(grid_size: tuple[int, int, int]) -> tuple[int, float]:
    width, height, max_iter = grid_size

    start_time = time.time()
    # Process all rows as a single task
    result = compute_mandelbrot_rows((0, height, width, height, max_iter))
    elapsed_time = time.time() - start_time

    return result, elapsed_time


def get_n_chunks(height: int, n_cores: int, chunking_factor: int) -> int:
    """Return the number of row chunks, capped at the number of rows."""
    return max(1, min(chunking_factor * n_cores, height))


def make_row_chunks(grid_size: tuple[int, int, int], n_chunks: int) -> list[tuple]:
    """Split the rows of the grid into n_chunks contiguous, balanced chunks.

    Leftover rows are spread across chunks, so chunk sizes differ by at
    most one row.

    Parameters
    ----------
    grid_size : tuple[int, int, int]
        Tuple of (width, height, max_iter)
    n_chunks : int
        Number of chunks to create

    Returns
    -------
    list[tuple]
        List of (start_row, end_row, width, height, max_iter) tuples
    """
    width, height, max_iter = grid_size
    return [
        (i * height // n_chunks, (i + 1) * height // n_chunks, width, height, max_iter)
        for i in range(n_chunks)
    ]


# moving docstring out of function for direct read into latex
# """Run computation in parallel by splitting rows across workers.

# Each worker processes a chunk of rows from the grid.

# Parameters
# ----------
# grid_size : tuple[int, int, int]
#     Tuple of (width, height, max_iter)
# ncores : int or None, default=None
#     Number of cores to use. If None, uses cpu_count - 1
# chunking_factor : int, default=1
#     Multiplier for number of chunks (n_chunks = chunking_factor * ncores)

# Returns
# -------
# tuple[int, float, int]
#     Tuple of (result sum, elapsed time in seconds, n_cores used)
# """


def run_parallel(
    grid_size: tuple[int, int, int], ncores: int | None = None, chunking_factor: int = 1
) -> tuple[int, float, int]:
    height = grid_size[1]

    if ncores is not None:
        n_cores = ncores
    else:
        n_cores = max(1, mp.cpu_count() - 1)

    n_chunks = get_n_chunks(height, n_cores, chunking_factor)
    chunks = make_row_chunks(grid_size, n_chunks)

    start_time = time.time()
    with mp.Pool(processes=n_cores) as pool:
        # chunksize=1 schedules each row chunk as its own task, so that
        # Pool.map does not add a second layer of batching
        results = pool.map(compute_mandelbrot_rows, chunks, chunksize=1)
    elapsed_time = time.time() - start_time

    # Sum results from all chunks
    total_result = sum(results)

    return total_result, elapsed_time, n_cores
