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
# # Parquet File Optimization: Impact of Grouping Structure
#
# This notebook demonstrates how the internal organization of Parquet files affects query performance. We'll explore:
#
# 1. **Row Groups**: How data is organized within a single Parquet file
# 2. **Partitioning**: How data is split across multiple files based on column values
# 3. **Performance Impact**: How matching grouping structure to query patterns improves performance
#
# ## Key Concepts
#
# - **Row Groups**: Internal chunks of data within a Parquet file. Statistics are stored per row group.
# - **Partitioning**: Physical file organization by column values (e.g., separate files per date)
# - **Predicate Pushdown**: Skipping irrelevant row groups/partitions based on query filters

# %%
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Create temporary directory for parquet files
temp_dir = Path("temp_parquet_demo")
temp_dir.mkdir(exist_ok=True)

print(f"PyArrow version: {pa.__version__}")
print(f"Pandas version: {pd.__version__}")

# %% [markdown]
# ## 1. Generate Sample Dataset
#
# We'll create a dataset with:
# - Multiple categories (e.g., regions, departments)
# - Temporal data (dates)
# - Numerical metrics
#
# This structure is common in analytics workloads where queries often filter by category or time range.

# %%
# Generate sample data
n_rows = 1_000_000
n_regions = 5
n_departments = 10
date_range = pd.date_range('2023-01-01', '2023-12-31', freq='D')

data = {
    'region': np.random.choice([f'Region_{i}' for i in range(n_regions)], n_rows),
    'department': np.random.choice([f'Dept_{i}' for i in range(n_departments)], n_rows),
    'date': np.random.choice(date_range, n_rows),
    'revenue': np.random.exponential(1000, n_rows),
    'quantity': np.random.poisson(50, n_rows),
    'customer_id': np.random.randint(1, 100000, n_rows),
    'transaction_id': range(n_rows)
}

df = pd.DataFrame(data)
df['year_month'] = df['date'].dt.to_period('M').astype(str)

print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)

# %% [markdown]
# ## 2. Write Parquet Files with Different Strategies
#
# We'll create three versions of the same dataset with different organization strategies:
#
# ### Strategy 1: No Grouping (Random Order)
# - Data written in random order
# - Row groups don't align with any logical grouping
# - Worst case for filtered queries
#
# ### Strategy 2: Sorted by Region
# - Data sorted by region before writing
# - Row groups contain data from similar regions
# - Good for queries filtering by region
#
# ### Strategy 3: Partitioned by Region
# - Data physically split into separate files per region
# - Best for queries filtering by region (entire files can be skipped)
#
# ### Strategy 4: Sorted by Date
# - Data sorted by date before writing
# - Good for time-range queries

# %%
# Strategy 1: No grouping (random order)
print("Writing Strategy 1: Random order...")
df_random = df.sample(frac=1, random_state=42).reset_index(drop=True)
file_random = temp_dir / "data_random.parquet"
df_random.to_parquet(
    file_random, 
    engine='pyarrow',
    row_group_size=50000  # Create multiple row groups
)

# Strategy 2: Sorted by region
print("Writing Strategy 2: Sorted by region...")
df_sorted_region = df.sort_values('region').reset_index(drop=True)
file_sorted_region = temp_dir / "data_sorted_region.parquet"
df_sorted_region.to_parquet(
    file_sorted_region,
    engine='pyarrow',
    row_group_size=50000
)

# Strategy 3: Partitioned by region
print("Writing Strategy 3: Partitioned by region...")
partition_dir = temp_dir / "data_partitioned"
if partition_dir.exists():
    shutil.rmtree(partition_dir)
df.to_parquet(
    partition_dir,
    engine='pyarrow',
    partition_cols=['region'],
    row_group_size=50000
)

# Strategy 4: Sorted by date
print("Writing Strategy 4: Sorted by date...")
df_sorted_date = df.sort_values('date').reset_index(drop=True)
file_sorted_date = temp_dir / "data_sorted_date.parquet"
df_sorted_date.to_parquet(
    file_sorted_date,
    engine='pyarrow',
    row_group_size=50000
)

print("\nFiles created:")
for f in temp_dir.glob("**/*.parquet"):
    size_mb = f.stat().st_size / 1024**2
    print(f"  {f.relative_to(temp_dir)}: {size_mb:.2f} MB")


# %% [markdown]
# ## 3. Inspect Parquet File Metadata
#
# Let's examine the row group structure to understand how data is organized.

# %%
def inspect_parquet_metadata(file_path):
    """Inspect row group metadata for a parquet file."""
    parquet_file = pq.ParquetFile(file_path)
    metadata = parquet_file.metadata
    
    print(f"\nFile: {file_path.name}")
    print(f"  Number of row groups: {metadata.num_row_groups}")
    print(f"  Total rows: {metadata.num_rows}")
    print(f"  Number of columns: {metadata.num_columns}")
    
    # Sample first few row groups to show region distribution
    print(f"\n  Row group details (first 3):")
    for i in range(min(3, metadata.num_row_groups)):
        rg = metadata.row_group(i)
        print(f"    Row group {i}: {rg.num_rows} rows")
        
        # Show statistics for region column if available
        for j in range(rg.num_columns):
            col = rg.column(j)
            if col.path_in_schema == 'region' and col.is_stats_set:
                try:
                    print(f"      Region stats: min={col.statistics.min}, max={col.statistics.max}")
                except:
                    print(f"      Region stats: available but not decoded")

# Inspect each strategy
inspect_parquet_metadata(file_random)
inspect_parquet_metadata(file_sorted_region)
inspect_parquet_metadata(file_sorted_date)


# %% [markdown]
# ## 4. Benchmark Query Performance
#
# We'll test different query patterns:
# 1. **Filter by single region** - Should benefit from region-sorted/partitioned data
# 2. **Filter by date range** - Should benefit from date-sorted data
# 3. **Filter by multiple columns** - Tests combined effects
# 4. **Aggregate query** - Full scan scenario

# %%
def benchmark_query(file_path, filters, n_runs=5, use_dataset=False):
    """
    Benchmark a query with filters.
    
    Parameters:
    -----------
    file_path : Path or str
        Path to parquet file or directory
    filters : list
        PyArrow filters for predicate pushdown
    n_runs : int
        Number of times to run the query
    use_dataset : bool
        Whether to use dataset API (for partitioned data)
    """
    times = []
    
    for _ in range(n_runs):
        start = time.time()
        
        if use_dataset:
            # Use dataset API for partitioned data
            dataset = pq.ParquetDataset(file_path, filters=filters)
            result = dataset.read_pandas().to_pandas()
        else:
            # Use ParquetFile for single files
            result = pq.read_table(file_path, filters=filters).to_pandas()
        
        elapsed = time.time() - start
        times.append(elapsed)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'rows_returned': len(result)
    }

print("Benchmark function defined. Ready to run queries.")

# %% [markdown]
# ### Query 1: Filter by Single Region
#
# This query filters for a single region. We expect:
# - **Random**: Slow (must scan all row groups)
# - **Sorted by region**: Fast (can skip many row groups using statistics)
# - **Partitioned by region**: Fastest (entire files can be skipped)
# - **Sorted by date**: Slow (region data scattered across row groups)

# %%
# Define filter for a single region
region_filter = [('region', '=', 'Region_0')]

print("Query 1: Filter by region = 'Region_0'\n")
print("=" * 60)

results_q1 = {}

print("\n1. Random order:")
results_q1['random'] = benchmark_query(file_random, region_filter)
print(f"   Mean time: {results_q1['random']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q1['random']['rows_returned']:,}")

print("\n2. Sorted by region:")
results_q1['sorted_region'] = benchmark_query(file_sorted_region, region_filter)
print(f"   Mean time: {results_q1['sorted_region']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q1['sorted_region']['rows_returned']:,}")

print("\n3. Partitioned by region:")
results_q1['partitioned'] = benchmark_query(partition_dir, region_filter, use_dataset=True)
print(f"   Mean time: {results_q1['partitioned']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q1['partitioned']['rows_returned']:,}")

print("\n4. Sorted by date:")
results_q1['sorted_date'] = benchmark_query(file_sorted_date, region_filter)
print(f"   Mean time: {results_q1['sorted_date']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q1['sorted_date']['rows_returned']:,}")

print("\n" + "=" * 60)
print(f"Speedup (sorted by region vs random): {results_q1['random']['mean_time'] / results_q1['sorted_region']['mean_time']:.2f}x")
print(f"Speedup (partitioned vs random): {results_q1['random']['mean_time'] / results_q1['partitioned']['mean_time']:.2f}x")

# %% [markdown]
# ### Query 2: Filter by Date Range
#
# This query filters for a specific date range. We expect:
# - **Random**: Slow
# - **Sorted by region**: Slow (dates scattered)
# - **Partitioned by region**: Slow (dates scattered across partitions)
# - **Sorted by date**: Fast (can skip row groups outside the range)

# %%
# Define filter for date range (one month)
date_filter = [
    ('date', '>=', pd.Timestamp('2023-01-01')),
    ('date', '<', pd.Timestamp('2023-02-01'))
]

print("Query 2: Filter by date range (January 2023)\n")
print("=" * 60)

results_q2 = {}

print("\n1. Random order:")
results_q2['random'] = benchmark_query(file_random, date_filter)
print(f"   Mean time: {results_q2['random']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q2['random']['rows_returned']:,}")

print("\n2. Sorted by region:")
results_q2['sorted_region'] = benchmark_query(file_sorted_region, date_filter)
print(f"   Mean time: {results_q2['sorted_region']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q2['sorted_region']['rows_returned']:,}")

print("\n3. Partitioned by region:")
results_q2['partitioned'] = benchmark_query(partition_dir, date_filter, use_dataset=True)
print(f"   Mean time: {results_q2['partitioned']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q2['partitioned']['rows_returned']:,}")

print("\n4. Sorted by date:")
results_q2['sorted_date'] = benchmark_query(file_sorted_date, date_filter)
print(f"   Mean time: {results_q2['sorted_date']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q2['sorted_date']['rows_returned']:,}")

print("\n" + "=" * 60)
print(f"Speedup (sorted by date vs random): {results_q2['random']['mean_time'] / results_q2['sorted_date']['mean_time']:.2f}x")
print(f"Speedup (sorted by date vs sorted by region): {results_q2['sorted_region']['mean_time'] / results_q2['sorted_date']['mean_time']:.2f}x")

# %% [markdown]
# ### Query 3: Combined Filters (Region + Date)
#
# This query uses both region and date filters. Performance depends on which column provides better filtering.

# %%
# Combined filter
combined_filter = [
    ('region', '=', 'Region_0'),
    ('date', '>=', pd.Timestamp('2023-01-01')),
    ('date', '<', pd.Timestamp('2023-02-01'))
]

print("Query 3: Combined filter (Region_0 + January 2023)\n")
print("=" * 60)

results_q3 = {}

print("\n1. Random order:")
results_q3['random'] = benchmark_query(file_random, combined_filter)
print(f"   Mean time: {results_q3['random']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q3['random']['rows_returned']:,}")

print("\n2. Sorted by region:")
results_q3['sorted_region'] = benchmark_query(file_sorted_region, combined_filter)
print(f"   Mean time: {results_q3['sorted_region']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q3['sorted_region']['rows_returned']:,}")

print("\n3. Partitioned by region:")
results_q3['partitioned'] = benchmark_query(partition_dir, combined_filter, use_dataset=True)
print(f"   Mean time: {results_q3['partitioned']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q3['partitioned']['rows_returned']:,}")

print("\n4. Sorted by date:")
results_q3['sorted_date'] = benchmark_query(file_sorted_date, combined_filter)
print(f"   Mean time: {results_q3['sorted_date']['mean_time']*1000:.2f} ms")
print(f"   Rows returned: {results_q3['sorted_date']['rows_returned']:,}")

print("\n" + "=" * 60)
print(f"Best strategy: Partitioned (region filter eliminates entire files)")

# %% [markdown]
# ## 5. Visualize Performance Comparison
#
# Let's create visualizations to clearly show the performance differences.

# %%
# Compile results into a DataFrame for visualization
results_data = []

for query_name, results in [('Region Filter', results_q1), 
                             ('Date Filter', results_q2), 
                             ('Combined Filter', results_q3)]:
    for strategy, data in results.items():
        results_data.append({
            'Query': query_name,
            'Strategy': strategy.replace('_', ' ').title(),
            'Mean Time (ms)': data['mean_time'] * 1000,
            'Std Time (ms)': data['std_time'] * 1000
        })

results_df = pd.DataFrame(results_data)

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

queries = ['Region Filter', 'Date Filter', 'Combined Filter']
colors = {'Random': '#e74c3c', 'Sorted Region': '#3498db', 
          'Partitioned': '#2ecc71', 'Sorted Date': '#f39c12'}

for idx, query in enumerate(queries):
    ax = axes[idx]
    query_data = results_df[results_df['Query'] == query]
    
    bars = ax.bar(range(len(query_data)), query_data['Mean Time (ms)'], 
                   color=[colors[s] for s in query_data['Strategy']])
    
    ax.set_xticks(range(len(query_data)))
    ax.set_xticklabels(query_data['Strategy'], rotation=45, ha='right')
    ax.set_ylabel('Query Time (ms)')
    ax.set_title(query)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, query_data['Mean Time (ms)'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.suptitle('Parquet Query Performance by Organization Strategy', 
             fontsize=14, y=1.02)
plt.show()

print("\nPerformance Summary:")
print(results_df.to_string(index=False))

# %% [markdown]
# ## 6. Key Takeaways and Best Practices
#
# ### Performance Results
#
# The benchmarks demonstrate that **aligning data organization with query patterns** significantly improves performance:
#
# 1. **Region-based queries benefit most from region-based organization**
#    - Sorted by region: Enables row group skipping via statistics
#    - Partitioned by region: Enables entire file skipping (best performance)
#
# 2. **Date-based queries benefit from date-based organization**
#    - Sorted by date: Enables efficient row group pruning for time ranges
#    - Random or region-organized data: Poor performance (dates scattered)
#
# 3. **Combined filters leverage the primary organization**
#    - Partitioning by the most selective filter provides best results
#    - Secondary filters are applied after primary pruning
#
# ### Best Practices
#
# 1. **Understand your query patterns**
#    - Profile which columns are most frequently used in WHERE clauses
#    - Consider query selectivity (how much data is filtered out)
#
# 2. **Choose organization strategy based on primary access pattern**
#    - **Partitioning**: Best for high-cardinality categorical columns that appear in most queries
#    - **Sorting**: Good for range queries or when partitioning would create too many files
#    - **Row group size**: Balance between granular statistics and file overhead
#
# 3. **Trade-offs to consider**
#    - Partitioning creates many files (storage overhead, metadata)
#    - Sorting helps but isn't as dramatic as partitioning
#    - Random organization forces full scans (only acceptable for small datasets or rare queries)
#
# 4. **Multi-dimensional access patterns**
#    - Can't optimize for all patterns simultaneously
#    - Consider creating multiple copies optimized for different workloads
#    - Or use columnar databases that support multi-dimensional indexing

# %%
# Cleanup temporary files
print("Cleaning up temporary files...")
shutil.rmtree(temp_dir)
print(f"Removed {temp_dir}")
