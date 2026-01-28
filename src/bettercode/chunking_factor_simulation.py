"""
Simulation to study the effect of chunking factor on parallel performance.

This script varies the chunking factor over [1, 10, 25, 50] with a fixed
image size of 2048x2048 to understand how work granularity affects
parallel efficiency.
"""

import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from bettercode.parallel_base import run_parallel, run_serial


def main():
    """Run chunking factor simulation and save results."""
    # Fixed grid size: 4096x4096 with 100 iterations
    grid_size = (4096, 4096, 100)
    width, height, max_iter = grid_size
    complexity = width * height * max_iter

    # Chunking factors to test
    chunking_factors = [1, 10, 25, 50, 100]

    # Test with different numbers of cores
    max_cores = mp.cpu_count()
    ncores_list = list(range(2, max(3, max_cores - 1), 2))

    print("Chunking Factor Simulation")
    print(f"Grid size: {width}x{height}, {max_iter} iterations")
    print(f"Available cores: {max_cores}")
    print(f"Testing with: {ncores_list} cores")
    print(f"Chunking factors: {chunking_factors}")
    print("=" * 70)

    # Run serial version once (baseline)
    print("\nRunning serial baseline...")
    result_serial, time_serial = run_serial(grid_size)
    print(f"Serial time: {time_serial:.2f} seconds")

    all_results = []

    # Add serial baseline to results
    all_results.append({
        'n_cores': 1,
        'chunking_factor': 1,
        'n_chunks': 1,
        'grid_width': width,
        'grid_height': height,
        'max_iterations': max_iter,
        'complexity': complexity,
        'serial_time': time_serial,
        'parallel_time': time_serial,
        'speedup': 1.0,
        'efficiency_percent': 100.0
    })

    # Run parallel versions with different chunking factors
    for chunking_factor in chunking_factors:
        print(f"\n{'='*70}")
        print(f"Chunking factor: {chunking_factor}")
        print(f"{'='*70}")

        for n_cores in ncores_list:
            n_chunks = chunking_factor * n_cores
            print(f"\n  {n_cores} cores, {n_chunks} chunks:")

            result_parallel, time_parallel, _ = run_parallel(
                grid_size, ncores=n_cores, chunking_factor=chunking_factor
            )

            speedup = time_serial / time_parallel
            efficiency = (speedup / n_cores) * 100

            print(f"    Time: {time_parallel:.2f}s, Speedup: {speedup:.2f}x, "
                  f"Efficiency: {efficiency:.1f}%")

            # Verify results match
            if result_serial != result_parallel:
                print("    WARNING: Results mismatch!")

            all_results.append({
                'n_cores': n_cores,
                'chunking_factor': chunking_factor,
                'n_chunks': n_chunks,
                'grid_width': width,
                'grid_height': height,
                'max_iterations': max_iter,
                'complexity': complexity,
                'serial_time': time_serial,
                'parallel_time': time_parallel,
                'speedup': speedup,
                'efficiency_percent': efficiency
            })

    # Save results to CSV
    df = pd.DataFrame(all_results)
    output_dir = Path(__file__).parent / 'data/parallel'
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'chunking_factor_simulation.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Set up output directory for figures
    output_dir_images = Path(__file__).parent / '../../../book/book/images'
    output_dir_images.mkdir(parents=True, exist_ok=True)

    # Figure 1: Speedup vs Number of Cores for each chunking factor
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    for cf in chunking_factors:
        df_cf = df[(df['chunking_factor'] == cf) & (df['n_cores'] > 1)]
        ax1.plot(df_cf['n_cores'], df_cf['speedup'],
                 'o-', linewidth=2, markersize=8, label=f'CF={cf}')

    # Add ideal scaling line
    ax1.plot(ncores_list, ncores_list, 'k--', alpha=0.5, label='Ideal scaling')
    ax1.axhline(y=1, color='red', linestyle=':', alpha=0.7, label='Serial baseline')

    ax1.set_xlabel('Number of Cores', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Speedup vs Cores by Chunking Factor', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Chunking Factor')

    plt.tight_layout()
    speedup_path = output_dir_images / 'chunking_factor_speedup.png'
    fig1.savefig(speedup_path, dpi=300, bbox_inches='tight')
    print(f"Speedup plot saved to: {speedup_path}")
    plt.close(fig1)

    # Figure 2: Efficiency vs Number of Cores for each chunking factor
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for cf in chunking_factors:
        df_cf = df[(df['chunking_factor'] == cf) & (df['n_cores'] > 1)]
        ax2.plot(df_cf['n_cores'], df_cf['efficiency_percent'],
                 'o-', linewidth=2, markersize=8, label=f'CF={cf}')

    ax2.axhline(y=100, color='red', linestyle=':', alpha=0.7, label='100% efficiency')

    ax2.set_xlabel('Number of Cores', fontsize=12)
    ax2.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax2.set_title('Efficiency vs Cores by Chunking Factor', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Chunking Factor')

    plt.tight_layout()
    efficiency_path = output_dir_images / 'chunking_factor_efficiency.png'
    fig2.savefig(efficiency_path, dpi=300, bbox_inches='tight')
    print(f"Efficiency plot saved to: {efficiency_path}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
