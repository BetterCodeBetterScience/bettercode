"""
Simple example of parallel processing for computationally intensive tasks.

This example demonstrates:
1. Using multiprocessing.Pool for parallel execution
2. Comparing serial vs parallel performance
3. How parallel advantage increases with computation difficulty

Uses Mandelbrot set calculation as a CPU-intensive, pure Python computation.
"""

import multiprocessing as mp
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bettercode.parallel_base import run_parallel, run_serial


def main(chunking_factor: int = 1) -> None:
    """Compare serial and parallel execution with varying difficulty.
    
    Parameters
    ----------
    chunking_factor : int, default=1
        Multiplier for number of chunks (n_chunks = chunking_factor * ncores)
    """
    # Test with different computational loads (powers of 2)
    # Format: (width, height, max_iterations)
    difficulty_levels = [
        (256, 256, 100),
        (512, 512, 100),
        (1024, 1024, 100),
        (2048, 2048, 100),
        (4096, 4096, 100),
        (8192, 8192, 100),
    ]
    
    # Test with different numbers of cores (2 to max_cores-2, in steps of 2)
    max_cores = mp.cpu_count()
    ncores_list = list(range(2, max(3, max_cores - 1), 2))
    
    print("Computing Mandelbrot set with parallel row processing")
    print(f"Available cores: {max_cores}")
    print(f"Chunking factor: {chunking_factor}")
    print(f"Testing with: {ncores_list} cores\n")
    print("=" * 70)

    # Run serial version once for each grid size
    print("\nRunning serial benchmarks...")
    print("-" * 70)
    serial_results = {}
    all_results = []
    for grid_size in difficulty_levels:
        width, height, max_iter = grid_size
        grid_desc = f"{width}x{height} grid, {max_iter} iterations"
        result_serial, time_serial = run_serial(grid_size)
        serial_results[grid_size] = (result_serial, time_serial)
        print(f"{grid_desc}: {time_serial:6.2f} seconds")

        # Add serial result to CSV
        complexity = width * height * max_iter
        all_results.append({
            'n_cores': 1,
            'chunking_factor': chunking_factor,
            'n_chunks': 1,
            'grid_width': width,
            'grid_height': height,
            'max_iterations': max_iter,
            'complexity': complexity,
            'log_complexity': int(complexity).bit_length(),
            'serial_time': time_serial,
            'parallel_time': time_serial,
            'speedup': 1.0,
            'efficiency_percent': 100.0
        })

    for n_cores in ncores_list:
        print(f"\n{'='*70}")
        print(f"Using {n_cores} cores")
        print(f"{'='*70}")

        for grid_size in difficulty_levels:
            width, height, max_iter = grid_size
            grid_desc = f"{width}x{height} grid, {max_iter} iterations"
            complexity = width * height * max_iter

            print(f"\n{grid_desc}:")
            print("-" * 70)

            # Get cached serial result
            result_serial, time_serial = serial_results[grid_size]
            print(f"Serial:   {time_serial:6.2f} seconds")

            # Run parallel version
            result_parallel, time_parallel, _ = run_parallel(grid_size, ncores=n_cores, chunking_factor=chunking_factor)
            n_chunks = chunking_factor * n_cores
            print(f"Parallel: {time_parallel:6.2f} seconds (split into {n_chunks} chunks)")

            # Calculate speedup
            speedup = time_serial / time_parallel
            efficiency = (speedup / n_cores) * 100
            print(f"Speedup:  {speedup:.2f}x ({efficiency:.1f}% parallel efficiency)")

            # Store for CSV
            all_results.append({
                'n_cores': n_cores,
                'chunking_factor': chunking_factor,
                'n_chunks': n_chunks,
                'grid_width': width,
                'grid_height': height,
                'max_iterations': max_iter,
                'complexity': complexity,
                'log_complexity': int(complexity).bit_length(),  # Log2 approximation
                'serial_time': time_serial,
                'parallel_time': time_parallel,
                'speedup': speedup,
                'efficiency_percent': efficiency
            })

            # Verify results match
            if result_serial == result_parallel:
                print("✓ Results match (deterministic)")
    
    print("\n" + "=" * 70)
    print("\nKey insight: As computation gets harder, parallel advantage increases!")
    
    # Save results to CSV
    df = pd.DataFrame(all_results)
    output_dir = Path(__file__).parent / 'data/parallel'
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'mandelbrot_scaling_cf{chunking_factor}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create plot with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: Speedup vs Log Complexity
    for n_cores in ncores_list:
        df_subset = df[df['n_cores'] == n_cores]
        ax1.plot(df_subset['log_complexity'], df_subset['speedup'], 
                'o-', linewidth=2, markersize=8, label=f'{n_cores} cores')
    
    # Add line at 1 to show serial performance baseline
    ax1.axhline(y=1, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='Serial baseline')
    
    # Add theoretical maximum lines
    for n_cores in ncores_list:
        ax1.axhline(y=n_cores, color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    ax1.set_xlabel('Log₂(Computational Complexity)', fontsize=12)
    ax1.set_ylabel('Speedup (Serial Time / Parallel Time)', fontsize=12)
    ax1.set_title('Parallel Processing Speedup', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Cores Used')
    
    # Panel 2: Efficiency vs Log Image Size (using log_complexity as proxy)
    for n_cores in ncores_list:
        df_subset = df[df['n_cores'] == n_cores]
        ax2.plot(df_subset['log_complexity'], df_subset['efficiency_percent'], 
                'o-', linewidth=2, markersize=8, label=f'{n_cores} cores')
    
    # Add 100% efficiency line as reference
    ax2.axhline(y=100, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='100% efficiency')
    
    ax2.set_xlabel('Log₂(Image Size)', fontsize=12)
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax2.set_title('Parallel Processing Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Cores Used')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / f'../../../book/book/images/mandelbrot_scaling_cf{chunking_factor}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        chunking_factor = int(sys.argv[1])
    else:
        chunking_factor = 1
    main(chunking_factor)

