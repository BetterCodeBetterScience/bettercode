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
# # Big Data demo
#
# This notebook demonstrates how we can use DuckDB or Polars to work with very large datasets.
#
# Here we will work with data from the NYC Taxi and Limousine Commission: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#
# This is a set of parquet files that total about 11 GB which represent individual taxi rides in new york city over the course of 10 years (about 752 million records).  
#
# my primary interest is computing the frequencies of rides between each taxi zone, separately per month/year.  
#

# %%
import os
import time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from bettercode.taxi_utils import (
    download_taxi_data,
    get_data_dirs,
)

load_dotenv()
DATADIR = Path(os.getenv("DATADIR"))
orig_dir, preproc_dir = get_data_dirs(DATADIR)

# %%
download_taxi_data(DATADIR, delay_between_downloads=30)

# %% [markdown]
# ## Preprocess data for consistent schemas
#
# The raw parquet files have inconsistent schemas across years (different datetime precisions, integer types, and column sets). We'll preprocess them once to create standardized files that both DuckDB and Polars can efficiently query.

# %%
from bettercode.taxi_utils import preprocess_all_files

# Preprocess files to standardize schemas (only runs once unless overwrite=True)
preprocess_all_files(DATADIR, overwrite=False)

# %% [markdown]
# First we will load all of the individual files and determine their size and memory usage

# %%
# Load all parquet files from orig_dir and add source filename column

data_files = list(orig_dir.glob("*.parquet"))
print(f"Found {len(data_files)} data files")


start_time = time.time()
# Load each file and add the source filename
memory = 0
rows = 0
for file_path in tqdm(sorted(data_files)):
    df = pd.read_parquet(file_path)
    memory += df.memory_usage(deep=True).sum()
    rows += len(df)

end_time = time.time()
print(f"\nTime to load files with pandas: {end_time - start_time:.2f} seconds")
print(f"\nTotal combined rows: {rows:,}")
print(f"Memory usage: {memory / 1024**3:.2f} GB")


# %% [markdown]
# ## Method 1: DuckDB Analysis
#
# DuckDB is designed for analytical queries on large datasets and can query Parquet files directly without loading them into memory. This is extremely efficient for datasets that don't fit in RAM.

# %% [markdown]
# ### DuckDB: Query all parquet files directly (no loading into memory)

# %%
# DuckDB can query parquet files directly without loading them into memory
# This is MUCH more memory efficient for large datasets

start_time = time.time()

# Query to compute zone-to-zone trip frequencies by month/year
# The preprocessed files already have PUBorough and DOBorough, so no joins needed!
query = f"""
SELECT 
    YEAR(tpep_pickup_datetime) as year,
    MONTH(tpep_pickup_datetime) as month,
    PULocationID as pickup_location_id,
    DOLocationID as dropoff_location_id,
    COUNT(*) as trip_count
FROM read_parquet('{preproc_dir}/*.parquet')
WHERE tpep_pickup_datetime IS NOT NULL
    AND YEAR(tpep_pickup_datetime) >= 2015
    AND YEAR(tpep_pickup_datetime) <= 2024
GROUP BY year, month, pickup_location_id, dropoff_location_id
ORDER BY year, month, pickup_location_id, dropoff_location_id
"""

# Execute query and get results
duckdb_results = duckdb.query(query).df()

end_time = time.time()

print(f"DuckDB analysis completed in {end_time - start_time:.2f} seconds")
print(f"Total route-month combinations: {len(duckdb_results):,}")
print(f"Memory usage of results: {duckdb_results.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Year range: {duckdb_results['year'].min()} to {duckdb_results['year'].max()}")


# %%
duckdb_results.head()

# %% [markdown]
# ### DuckDB: Example queries on the results

# %%
# Find the most popular routes for a specific month
print("Top 10 zone-to-zone routes in January 2024:")
jan_2024 = duckdb_results[
    (duckdb_results['year'] == 2024) & (duckdb_results['month'] == 1)
].nlargest(10, 'trip_count')[['pickup_location_id', 'dropoff_location_id', 'trip_count']]
print(jan_2024)

print("\n" + "="*60 + "\n")

# Total trips by year
print("Total trips by year:")
yearly_totals = duckdb.query("""
    SELECT year, SUM(trip_count) as total_trips
    FROM duckdb_results
    GROUP BY year
    ORDER BY year
""").df()
print(yearly_totals)

# %% [markdown]
# ### DuckDB: Create a pivot table for a specific month

# %%
# Create a zone-to-zone matrix for January 2024
jan_2024_data = duckdb_results[
    (duckdb_results['year'] == 2024) & (duckdb_results['month'] == 1)
]

zone_matrix = jan_2024_data.pivot(
    index='pickup_location_id',
    columns='dropoff_location_id', 
    values='trip_count'
).fillna(0).astype(int)

print(f"Zone-to-zone trip matrix for January 2024 (shape: {zone_matrix.shape}):")
print(zone_matrix.iloc[:10, :10])  # Show first 10x10 subset

# %% [markdown]
# ## Method 2: Polars Analysis
#
# Polars is a DataFrame library designed for speed and efficiency. It uses lazy evaluation and can process data in parallel. Like DuckDB, it can read Parquet files efficiently.

# %%
# Import polars
import polars as pl

print(f"Polars version: {pl.__version__}")

# %% [markdown]
# ### Polars: Lazy scan preprocessed parquet files
#
# With preprocessed files that have consistent schemas, Polars can efficiently scan all files with its clean API and lazy evaluation.

# %%
start_time = time.time()

# For this large dataset with Polars, we need to be very careful about memory
# Strategy: Process and aggregate without joins, use streaming mode
trips_lazy = pl.scan_parquet(str(preproc_dir / "*.parquet"))

# Streaming aggregation: filter, extract date parts, group by, aggregate
# This should use minimal memory by processing in chunks
polars_results = (
    trips_lazy
    .select([
        pl.col("tpep_pickup_datetime"),
        pl.col("PULocationID"),
        pl.col("DOLocationID")
    ])
    .filter(pl.col("tpep_pickup_datetime").is_not_null())
    .with_columns([
        pl.col("tpep_pickup_datetime").dt.year().alias("year"),
        pl.col("tpep_pickup_datetime").dt.month().alias("month"),
    ])
    .filter(
        (pl.col("year") >= 2015) & 
        (pl.col("year") <= 2024) 
    )
    .select(["year", "month", "PULocationID", "DOLocationID"])
    .group_by(["year", "month", "PULocationID", "DOLocationID"])
    .len()
    .sort(["year", "month", "PULocationID", "DOLocationID"])
    .collect(engine="streaming")  # Use streaming engine for memory efficiency
)

# Rename and add zone names to the aggregated result
polars_results = (
    polars_results
    .rename({"len": "trip_count"})
)

end_time = time.time()

print(f"Polars analysis completed in {end_time - start_time:.2f} seconds")
print(f"Total route-month combinations: {len(polars_results):,}")
print(f"Memory usage of results: {polars_results.estimated_size() / 1024**2:.2f} MB")
print(f"Year range: {polars_results['year'].min()} to {polars_results['year'].max()}")
polars_results.head(10)

# %% [markdown]
# ### Polars: Example queries on the results

# %%
# Find the most popular routes for January 2024
print("Top 10 zone-to-zone routes in January 2024 (Polars):")
jan_2024_polars = (
    polars_results
    .filter((pl.col("year") == 2024) & (pl.col("month") == 1))
    .sort("trip_count", descending=True)
    .select(["PULocationID",  "DOLocationID",  "trip_count"])
    .head(10)
)
print(jan_2024_polars)

print("\n" + "="*60 + "\n")

# Total trips by year
print("Total trips by year (Polars):")
yearly_totals_polars = (
    polars_results
    .group_by("year")
    .agg(pl.col("trip_count").sum().alias("total_trips"))
    .sort("year")
)
print(yearly_totals_polars)

# %%

# %% [markdown]
# ### Verify DuckDB and Polars Results Match
#
# Let's compare the results from both methods to ensure they're computing the same values.

# %%
# Convert Polars results to Pandas for comparison
polars_df = polars_results.to_pandas()

# Sort both dataframes the same way for comparison
duckdb_sorted = duckdb_results.sort_values(['year', 'month', 'pickup_location_id', 'dropoff_location_id']).reset_index(drop=True)
polars_sorted = polars_df.sort_values(['year', 'month', 'PULocationID', 'DOLocationID']).reset_index(drop=True)

# Rename Polars columns to match DuckDB
polars_sorted = polars_sorted.rename(columns={
    'PULocationID': 'pickup_location_id',
    'DOLocationID': 'dropoff_location_id'
})

# Compare the key columns (year, month, location IDs, trip_count)
cols_to_compare = ['year', 'month', 'pickup_location_id', 'dropoff_location_id', 'trip_count']

print("Comparing DuckDB and Polars results:")
print(f"DuckDB rows: {len(duckdb_sorted):,}")
print(f"Polars rows: {len(polars_sorted):,}")
print(f"Rows match: {len(duckdb_sorted) == len(polars_sorted)}")

# Check if values match
if len(duckdb_sorted) == len(polars_sorted):
    comparison = duckdb_sorted[cols_to_compare].equals(polars_sorted[cols_to_compare])
    print(f"\nAll values match: {comparison}")
    
    if not comparison:
        # Find differences
        mask = (duckdb_sorted[cols_to_compare] != polars_sorted[cols_to_compare]).any(axis=1)
        differences = duckdb_sorted[mask][cols_to_compare]
        print(f"\nFound {len(differences)} differing rows:")
        print(differences.head(10))
    else:
        print("\n✓ Results are identical!")
        
    # Compare totals
    duckdb_total = duckdb_sorted['trip_count'].sum()
    polars_total = polars_sorted['trip_count'].sum()
    print(f"\nTotal trips (DuckDB): {duckdb_total:,}")
    print(f"Total trips (Polars): {polars_total:,}")
    print(f"Difference: {abs(duckdb_total - polars_total):,}")
else:
    print("\n⚠ Different number of rows - investigating...")
    # Show which combinations are in one but not the other
    duckdb_keys = set(zip(duckdb_sorted['year'], duckdb_sorted['month'], 
                          duckdb_sorted['pickup_location_id'], duckdb_sorted['dropoff_location_id']))
    polars_keys = set(zip(polars_sorted['year'], polars_sorted['month'],
                          polars_sorted['pickup_location_id'], polars_sorted['dropoff_location_id']))
    
    only_duckdb = duckdb_keys - polars_keys
    only_polars = polars_keys - duckdb_keys
    
    print(f"Combinations only in DuckDB: {len(only_duckdb)}")
    print(f"Combinations only in Polars: {len(only_polars)}")

# %% [markdown]
# ### Polars: Create a pivot table for a specific month

# %%
# Create a zone-to-zone matrix for January 2024 using Polars pivot
jan_2024_polars_data = polars_results.filter(
    (pl.col("year") == 2024) & (pl.col("month") == 1)
)

# Polars native pivot
zone_matrix_polars = jan_2024_polars_data.pivot(
    values="trip_count",
    index="PULocationID",
    columns="DOLocationID",
    aggregate_function="sum"
).fill_null(0)

print(f"Zone-to-zone trip matrix for January 2024 (Polars) - shape: {zone_matrix_polars.shape}:")
print(zone_matrix_polars.head(10).select(zone_matrix_polars.columns[:11]))  # Show first 10 rows and 11 columns

# %% [markdown]
# ## Comparison: DuckDB vs Polars vs Pandas
#
# Let's compare the three approaches for working with large datasets:

# %% [markdown]
# ## Method 3: Dask Analysis
#
# Dask is a parallel computing library that provides familiar pandas-like APIs with lazy evaluation. It can handle datasets larger than memory by processing data in chunks and can scale from a single machine to a cluster.

# %%
# Import dask
import dask
import dask.dataframe as dd

print(f"Dask version: {dask.__version__}")

# %% [markdown]
# ### Dask: Lazy load preprocessed parquet files
#
# Dask uses lazy evaluation similar to Polars. Operations build up a task graph that is only executed when you call `.compute()`. This allows Dask to optimize the computation and process data in chunks.

# %%
start_time = time.time()

# Read all parquet files lazily
trips_dask = dd.read_parquet(preproc_dir / "*.parquet")

# Filter, extract date parts, and aggregate
# Dask uses pandas-like API with lazy evaluation
dask_results = (
    trips_dask
    .dropna(subset=["tpep_pickup_datetime"])
    .assign(
        year=lambda df: df["tpep_pickup_datetime"].dt.year,
        month=lambda df: df["tpep_pickup_datetime"].dt.month,
    )
    .query("year >= 2015 and year <= 2024")
    .groupby(["year", "month", "PULocationID", "DOLocationID"])
    .size()
    .reset_index()
    .rename(columns={0: "trip_count"})
    .compute()  # Execute the computation
)

# Sort results
dask_results = dask_results.sort_values(
    ["year", "month", "PULocationID", "DOLocationID"]
).reset_index(drop=True)

end_time = time.time()

print(f"Dask analysis completed in {end_time - start_time:.2f} seconds")
print(f"Total route-month combinations: {len(dask_results):,}")
print(f"Memory usage of results: {dask_results.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Year range: {dask_results['year'].min()} to {dask_results['year'].max()}")
dask_results.head(10)

# %% [markdown]
# ### Dask: Example queries on the results

# %%
# Find the most popular routes for January 2024
print("Top 10 zone-to-zone routes in January 2024 (Dask):")
jan_2024_dask = (
    dask_results
    .query("year == 2024 and month == 1")
    .nlargest(10, "trip_count")[["PULocationID", "DOLocationID", "trip_count"]]
)
print(jan_2024_dask)

print("\n" + "="*60 + "\n")

# Total trips by year
print("Total trips by year (Dask):")
yearly_totals_dask = (
    dask_results
    .groupby("year")["trip_count"]
    .sum()
    .reset_index()
    .rename(columns={"trip_count": "total_trips"})
    .sort_values("year")
)
print(yearly_totals_dask)

# %% [markdown]
# ### Dask: Create a pivot table for a specific month

# %%
# Create a zone-to-zone matrix for January 2024 using Dask results
# Note: dask_results is already a pandas DataFrame after .compute()
jan_2024_dask_data = dask_results.query("year == 2024 and month == 1")

zone_matrix_dask = jan_2024_dask_data.pivot(
    index="PULocationID",
    columns="DOLocationID",
    values="trip_count"
).fillna(0).astype(int)

print(f"Zone-to-zone trip matrix for January 2024 (Dask) - shape: {zone_matrix_dask.shape}:")
print(zone_matrix_dask.iloc[:10, :10])  # Show first 10x10 subset

# %% [markdown]
# ### Key Differences:
#
# **DuckDB:**
# - SQL-based interface (familiar for SQL users)
# - Queries parquet files directly without loading into memory
# - Excellent for analytical queries on data that doesn't fit in RAM
# - Can create persistent databases for repeated queries
# - Best for: SQL users, datasets larger than RAM, persistent storage
#
# **Polars:**
# - DataFrame API (similar to Pandas but faster)
# - Uses lazy evaluation for query optimization
# - Parallel processing by default
# - Memory-efficient with columnar storage
# - Best for: DataFrame users, data pipeline processing, speed-critical applications
#
# **Dask:**
# - Pandas-like API (easiest transition from pandas)
# - Uses lazy evaluation with task graphs
# - Scales from single machine to distributed clusters
# - Integrates well with the Python ecosystem (NumPy, scikit-learn)
# - Best for: Pandas users, scaling existing code, distributed computing
#
# **Pandas (from earlier cell):**
# - Traditional DataFrame API (most familiar)
# - Loads entire dataset into memory
# - Single-threaded processing
# - Best for: Smaller datasets that fit in RAM, quick prototyping
