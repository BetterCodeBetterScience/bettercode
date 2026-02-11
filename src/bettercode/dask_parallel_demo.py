"""Dask Distributed Scheduler Demo.

This script demonstrates Dask's parallel processing capabilities on the full
NYC Taxi dataset (~750 million records, ~11GB on disk) using the Distributed
Scheduler.

The Distributed Scheduler provides:
- Dashboard for real-time monitoring (http://localhost:8787)
- Proper memory management with spilling to disk
- Worker isolation and automatic recovery

Usage:
    python -m bettercode.dask_parallel_demo

    # With custom data directory:
    DATADIR=/path/to/data python -m bettercode.dask_parallel_demo

Note on "unmanaged memory" warnings:
    These warnings are often benign on macOS. The memory allocator doesn't
    always return freed memory to the OS immediately. For better memory
    behavior, install jemalloc: brew install jemalloc
    Then run with: DYLD_INSERT_LIBRARIES=$(brew --prefix jemalloc)/lib/libjemalloc.dylib python -m bettercode.dask_parallel_demo
"""

import gc
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil
from distributed import Client, LocalCluster
from dotenv import load_dotenv

from bettercode.taxi_utils import (
    download_taxi_data,
    get_data_dirs,
    preprocess_all_files,
)

# Configure logging - suppress distributed warnings for cleaner output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce noise from distributed scheduler
logging.getLogger("distributed.worker.memory").setLevel(logging.ERROR)
logging.getLogger("distributed.nanny.memory").setLevel(logging.ERROR)
logging.getLogger("distributed.worker").setLevel(logging.ERROR)
logging.getLogger("distributed.scheduler").setLevel(logging.WARNING)


def get_cluster_config() -> dict:
    """Calculate optimal cluster configuration for distributed scheduler.

    Key principles for single-machine distributed:
    - Fewer workers with more memory each (reduces inter-worker communication)
    - Single thread per worker (prevents concurrent memory spikes)
    - Generous memory limits (allows room for spilling)

    Returns:
        Dictionary with cluster configuration.
    """
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = psutil.cpu_count()

    # Use 60% of available memory for workers
    # Leave 40% for OS, spilling buffers, and headroom
    usable_memory_gb = available_memory_gb * 0.6

    # Fewer workers = less inter-process communication overhead
    # 2-4 workers is optimal for single-machine distributed
    n_workers = min(4, max(2, cpu_count // 4))

    # Generous memory per worker
    memory_per_worker_gb = usable_memory_gb / n_workers
    memory_limit = f"{int(memory_per_worker_gb)}GiB"

    # Single thread per worker to prevent concurrent memory spikes
    threads_per_worker = 1

    logger.info(f"System: {total_memory_gb:.1f}GB RAM, {cpu_count} CPUs")
    logger.info(f"Available: {available_memory_gb:.1f}GB RAM")
    logger.info(f"Cluster: {n_workers} workers × {threads_per_worker} thread, {memory_limit} each")

    return {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "memory_limit": memory_limit,
    }


def compute_trip_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trip statistics per partition.

    Returns AGGREGATED results (small output) not row-level features.
    This is key for memory efficiency - input is large, output is small.
    """
    duration_minutes = (
        (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
        / 60
    )

    valid = (
        (duration_minutes > 0) & (duration_minutes < 180) & (df["trip_distance"] > 0)
    )

    if valid.sum() == 0:
        return pd.DataFrame()

    duration_valid = duration_minutes[valid]
    distance_valid = df.loc[valid, "trip_distance"]
    fare_valid = df.loc[valid, "fare_amount"]
    tip_valid = df.loc[valid, "tip_amount"]

    speed_mph = distance_valid / (duration_valid / 60)
    fare_per_mile = fare_valid / distance_valid
    tip_pct = (tip_valid / fare_valid * 100).replace([np.inf, -np.inf], np.nan)

    return pd.DataFrame(
        [
            {
                "n_valid_trips": valid.sum(),
                "mean_duration": duration_valid.mean(),
                "mean_speed": speed_mph.mean(),
                "median_speed": speed_mph.median(),
                "mean_fare_per_mile": fare_per_mile.mean(),
                "mean_tip_pct": tip_pct.mean(),
                "pct_tipped": (tip_valid > 0).mean() * 100,
            }
        ]
    )


def bootstrap_zone_ci(zone_data: pd.DataFrame, n_bootstrap: int = 1000) -> dict | None:
    """Compute bootstrap CI for mean daily trips in a zone."""
    zone_id = zone_data["PULocationID"].iloc[0]
    trips = zone_data["trip_count"].values

    if len(trips) < 30:
        return None

    rng = np.random.default_rng(int(zone_id))
    boot_means = [
        rng.choice(trips, size=len(trips), replace=True).mean()
        for _ in range(n_bootstrap)
    ]

    return {
        "PULocationID": zone_id,
        "n_days": len(trips),
        "mean_daily_trips": trips.mean(),
        "ci_lower": np.percentile(boot_means, 2.5),
        "ci_upper": np.percentile(boot_means, 97.5),
    }


def run_gc_on_workers(client: Client) -> None:
    """Run garbage collection on all workers to free memory."""
    client.run(gc.collect)


def example_streaming_aggregations(client: Client, ddf: dd.DataFrame) -> dict:
    """Example 1: Streaming aggregations on 750M+ rows."""
    logger.info("=" * 60)
    logger.info("Example 1: Streaming Aggregations on Full Dataset")
    logger.info("=" * 60)

    start_time = time.time()

    stats = {
        "trip_distance_count": ddf["trip_distance"].count(),
        "trip_distance_mean": ddf["trip_distance"].mean(),
        "trip_distance_std": ddf["trip_distance"].std(),
        "trip_distance_min": ddf["trip_distance"].min(),
        "trip_distance_max": ddf["trip_distance"].max(),
        "fare_amount_mean": ddf["fare_amount"].mean(),
        "fare_amount_std": ddf["fare_amount"].std(),
        "tip_amount_mean": ddf["tip_amount"].mean(),
        "total_amount_mean": ddf["total_amount"].mean(),
    }

    results = dask.compute(stats)[0]

    elapsed = time.time() - start_time
    total_rows = int(results["trip_distance_count"])

    logger.info(f"Processed {total_rows:,} rows in {elapsed:.2f} seconds")
    logger.info(f"Throughput: {total_rows / elapsed / 1e6:.1f} million rows/second")
    logger.info("Statistics:")
    for k, v in results.items():
        if "count" in k:
            logger.info(f"  {k}: {v:,}")
        else:
            logger.info(f"  {k}: {v:,.2f}")

    # Clean up worker memory
    run_gc_on_workers(client)

    return results


def example_map_partitions(client: Client, ddf: dd.DataFrame) -> dict:
    """Example 2: Custom feature engineering with map_partitions."""
    logger.info("=" * 60)
    logger.info("Example 2: Custom Feature Engineering with map_partitions")
    logger.info("=" * 60)

    start_time = time.time()

    meta = pd.DataFrame(
        {
            "n_valid_trips": pd.Series(dtype="int64"),
            "mean_duration": pd.Series(dtype="float64"),
            "mean_speed": pd.Series(dtype="float64"),
            "median_speed": pd.Series(dtype="float64"),
            "mean_fare_per_mile": pd.Series(dtype="float64"),
            "mean_tip_pct": pd.Series(dtype="float64"),
            "pct_tipped": pd.Series(dtype="float64"),
        }
    )

    partition_stats = ddf.map_partitions(compute_trip_stats, meta=meta).compute()

    total_trips = partition_stats["n_valid_trips"].sum()
    weights = partition_stats["n_valid_trips"] / total_trips

    combined_stats = {
        "total_valid_trips": total_trips,
        "mean_duration_min": (partition_stats["mean_duration"] * weights).sum(),
        "mean_speed_mph": (partition_stats["mean_speed"] * weights).sum(),
        "mean_fare_per_mile": (partition_stats["mean_fare_per_mile"] * weights).sum(),
        "mean_tip_pct": (partition_stats["mean_tip_pct"] * weights).sum(),
        "pct_tipped": (partition_stats["pct_tipped"] * weights).sum(),
    }

    elapsed = time.time() - start_time

    logger.info(f"Processed {total_trips:,.0f} valid trips in {elapsed:.2f} seconds")
    logger.info("Combined Statistics:")
    for k, v in combined_stats.items():
        if "trips" in k:
            logger.info(f"  {k}: {v:,.0f}")
        else:
            logger.info(f"  {k}: {v:.2f}")

    run_gc_on_workers(client)

    return combined_stats


def example_groupby_aggregations(client: Client, ddf: dd.DataFrame) -> pd.DataFrame:
    """Example 3: GroupBy aggregations at scale."""
    logger.info("=" * 60)
    logger.info("Example 3: GroupBy Aggregations at Scale")
    logger.info("=" * 60)

    start_time = time.time()

    location_stats = (
        ddf.groupby("PULocationID")
        .agg(
            {
                "trip_distance": ["count", "mean"],
                "fare_amount": "mean",
                "tip_amount": "mean",
            }
        )
        .compute()
    )

    location_stats.columns = ["_".join(col).strip() for col in location_stats.columns]
    location_stats = location_stats.rename(columns={"trip_distance_count": "trip_count"})

    elapsed = time.time() - start_time

    logger.info(f"GroupBy completed in {elapsed:.2f} seconds")
    logger.info(f"Found {len(location_stats)} unique pickup locations")
    logger.info(f"Total trips: {location_stats['trip_count'].sum():,}")
    logger.info("Top 10 busiest pickup locations:")
    top_10 = location_stats.nlargest(10, "trip_count")[
        ["trip_count", "trip_distance_mean", "fare_amount_mean"]
    ]
    logger.info(f"\n{top_10.to_string()}")

    run_gc_on_workers(client)

    return location_stats


def example_time_series(client: Client, ddf: dd.DataFrame) -> pd.DataFrame:
    """Example 4: Time series aggregation using map_partitions to avoid shuffle.

    Shuffle operations (groupby across all data) are memory-intensive.
    For time series, we can aggregate per-partition first, then combine.
    """
    logger.info("=" * 60)
    logger.info("Example 4: Time Series Aggregation (shuffle-free)")
    logger.info("=" * 60)

    start_time = time.time()

    def partition_monthly_stats(df):
        """Aggregate monthly stats within a single partition."""
        df = df.copy()
        df["year"] = df["tpep_pickup_datetime"].dt.year
        df["month"] = df["tpep_pickup_datetime"].dt.month
        result = (
            df.groupby(["year", "month"])
            .agg({
                "trip_distance": ["count", "sum"],
                "fare_amount": "sum",
                "total_amount": "sum",
            })
        )
        # Flatten MultiIndex columns
        result.columns = ["trip_count", "trip_distance_sum", "fare_sum", "total_revenue"]
        return result.reset_index()

    # Define output schema (with flattened columns)
    meta = pd.DataFrame({
        "year": pd.Series(dtype="int32"),
        "month": pd.Series(dtype="int32"),
        "trip_count": pd.Series(dtype="int64"),
        "trip_distance_sum": pd.Series(dtype="float64"),
        "fare_sum": pd.Series(dtype="float64"),
        "total_revenue": pd.Series(dtype="float64"),
    })

    # Aggregate per partition (no shuffle needed)
    partition_stats = ddf.map_partitions(partition_monthly_stats, meta=meta).compute()

    # Combine partition results
    monthly_stats = (
        partition_stats.groupby(["year", "month"])
        .agg({
            "trip_count": "sum",
            "trip_distance_sum": "sum",
            "fare_sum": "sum",
            "total_revenue": "sum",
        })
        .reset_index()
    )

    # Compute averages
    monthly_stats["avg_distance"] = monthly_stats["trip_distance_sum"] / monthly_stats["trip_count"]
    monthly_stats["avg_fare"] = monthly_stats["fare_sum"] / monthly_stats["trip_count"]
    monthly_stats = monthly_stats.sort_values(["year", "month"])

    elapsed = time.time() - start_time

    # Filter to reasonable years
    valid_years = monthly_stats[
        (monthly_stats["year"] >= 2009) & (monthly_stats["year"] <= 2024)
    ]

    logger.info(f"Time series aggregation completed in {elapsed:.2f} seconds")
    logger.info(f"Monthly statistics ({len(valid_years)} months with valid data)")

    run_gc_on_workers(client)

    return monthly_stats


def example_bootstrap(client: Client, ddf: dd.DataFrame) -> pd.DataFrame:
    """Example 5: Parallel bootstrap using distributed client.map."""
    logger.info("=" * 60)
    logger.info("Example 5: Parallel Bootstrap with Distributed")
    logger.info("=" * 60)

    # Step 1: Aggregate daily statistics per zone (shuffle-free approach)
    logger.info("Step 1: Aggregating daily statistics per zone...")
    start_time = time.time()

    def partition_daily_counts(df):
        """Count trips per zone per day within a partition."""
        df = df.copy()
        df["date"] = df["tpep_pickup_datetime"].dt.date
        return df.groupby(["PULocationID", "date"]).size().reset_index(name="trip_count")

    meta = pd.DataFrame({
        "PULocationID": pd.Series(dtype="int64"),
        "date": pd.Series(dtype="object"),
        "trip_count": pd.Series(dtype="int64"),
    })

    # Aggregate per partition, then combine
    partition_counts = ddf.map_partitions(partition_daily_counts, meta=meta).compute()
    daily_zone_stats = (
        partition_counts.groupby(["PULocationID", "date"])["trip_count"]
        .sum()
        .reset_index()
    )

    agg_time = time.time() - start_time
    logger.info(f"Aggregation completed in {agg_time:.2f} seconds")
    logger.info(f"Reduced to {len(daily_zone_stats):,} daily zone records")

    run_gc_on_workers(client)

    # Step 2: Run parallel bootstrap using client.map
    logger.info("Step 2: Running parallel bootstrap across workers...")
    start_time = time.time()

    zone_groups = [group for _, group in daily_zone_stats.groupby("PULocationID")]

    # Use client.map for true distributed parallelism
    futures = client.map(bootstrap_zone_ci, zone_groups)
    results = client.gather(futures)

    bootstrap_results = pd.DataFrame([r for r in results if r is not None])

    boot_time = time.time() - start_time

    logger.info(f"Bootstrap completed in {boot_time:.2f} seconds")
    logger.info(f"Computed CIs for {len(bootstrap_results)} zones")
    logger.info("\nTop 10 zones by mean daily trips:")
    top_zones = bootstrap_results.nlargest(10, "mean_daily_trips")
    logger.info(f"\n{top_zones.to_string(index=False)}")

    run_gc_on_workers(client)

    return bootstrap_results


def example_comparison(client: Client, ddf: dd.DataFrame, preproc_dir: Path) -> None:
    """Example 6: Performance comparison with Polars."""
    logger.info("=" * 60)
    logger.info("Example 6: Performance Comparison - Dask vs Polars")
    logger.info("=" * 60)

    # Dask distributed
    logger.info("Running Dask Distributed...")
    start_time = time.time()
    _ = ddf.groupby("PULocationID")["trip_distance"].mean().compute()
    dask_time = time.time() - start_time
    logger.info(f"Dask Distributed: {dask_time:.2f} seconds")

    run_gc_on_workers(client)

    # Polars lazy
    try:
        import polars as pl

        logger.info("Running Polars (lazy)...")
        start_time = time.time()
        _ = (
            pl.scan_parquet(str(preproc_dir / "*.parquet"))
            .group_by("PULocationID")
            .agg(pl.col("trip_distance").mean().alias("mean_trip_distance"))
            .collect()
        )
        polars_time = time.time() - start_time
        logger.info(f"Polars (lazy): {polars_time:.2f} seconds")

        if polars_time < dask_time:
            logger.info(f"\nPolars is {dask_time / polars_time:.2f}x faster than Dask")
        else:
            logger.info(f"\nDask is {polars_time / dask_time:.2f}x faster than Polars")
    except ImportError:
        logger.warning("Polars not installed, skipping comparison")


def main():
    """Main entry point for the Dask distributed demo."""
    logger.info("=" * 60)
    logger.info("Dask Distributed Scheduler Demo")
    logger.info("=" * 60)

    # Load environment and configure paths
    load_dotenv()
    datadir_env = os.getenv("DATADIR")
    if not datadir_env:
        logger.error("DATADIR environment variable not set")
        sys.exit(1)

    datadir = Path(datadir_env)
    orig_dir, preproc_dir = get_data_dirs(datadir)

    # Download and preprocess data if needed
    logger.info("Checking data availability...")
    download_taxi_data(datadir, delay_between_downloads=30)
    preprocess_all_files(datadir, overwrite=False)

    # Get cluster configuration
    config = get_cluster_config()

    # Configure distributed scheduler memory management
    # See: https://distributed.dask.org/en/stable/worker-memory.html
    dask.config.set({
        # Memory thresholds (fraction of worker memory limit)
        "distributed.worker.memory.target": 0.60,      # Start spilling at 60%
        "distributed.worker.memory.spill": 0.70,       # Spill more at 70%
        "distributed.worker.memory.pause": 0.80,       # Pause worker at 80%
        "distributed.worker.memory.terminate": 0.95,   # Kill worker at 95%
        # Don't count recent memory as "unmanaged" (reduces false warnings)
        "distributed.worker.memory.recent-to-old-time": "30s",
        # Allow some tolerance for memory spikes
        "distributed.worker.memory.rebalance.measure": "managed",
    })

    # Create a local directory for spilling
    spill_dir = Path(tempfile.gettempdir()) / "dask-spill"
    spill_dir.mkdir(exist_ok=True)

    logger.info("Starting Dask distributed cluster...")
    cluster = LocalCluster(
        n_workers=config["n_workers"],
        threads_per_worker=config["threads_per_worker"],
        memory_limit=config["memory_limit"],
        local_directory=str(spill_dir),
    )

    client = Client(cluster)

    logger.info(f"Dashboard: {client.dashboard_link}")
    logger.info(f"Spill directory: {spill_dir}")

    try:
        # Load data lazily
        logger.info("Loading parquet files lazily...")
        ddf = dd.read_parquet(preproc_dir / "*.parquet")
        logger.info(f"Number of partitions: {ddf.npartitions}")
        logger.info(f"Columns: {list(ddf.columns)}")

        # Run examples
        example_streaming_aggregations(client, ddf)
        example_map_partitions(client, ddf)
        example_groupby_aggregations(client, ddf)
        example_time_series(client, ddf)
        example_bootstrap(client, ddf)
        example_comparison(client, ddf, preproc_dir)

        logger.info("=" * 60)
        logger.info("Demo completed successfully!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        logger.info("Shutting down cluster...")
        client.close()
        cluster.close()
        logger.info("Done")


if __name__ == "__main__":
    main()
