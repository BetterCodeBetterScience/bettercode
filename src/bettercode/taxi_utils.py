"""Utility functions for NYC taxi data analysis.

This module provides functions to download, preprocess, and analyze
NYC taxi trip data, comparing different data access methods.
"""

import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm


BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def get_data_dirs(datadir: Path) -> tuple[Path, Path]:
    """Return (orig_dir, preproc_dir) paths, creating them if needed."""
    orig_dir = datadir / "nyctaxi" / "orig"
    preproc_dir = datadir / "nyctaxi" / "preproc"
    orig_dir.mkdir(parents=True, exist_ok=True)
    preproc_dir.mkdir(parents=True, exist_ok=True)
    return orig_dir, preproc_dir


def generate_file_urls(
    start_year: int = 2015, end_year: int = 2024
) -> list[tuple[str, str]]:
    """Generate list of (url, filename) tuples for all monthly parquet files."""
    urls = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url = BASE_URL.format(year=year, month=month)
            filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
            urls.append((url, filename))
    return urls


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file if it doesn't exist. Return True if downloaded."""
    if dest_path.exists():
        return False
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with (
        open(dest_path, "wb") as f,
        tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=dest_path.name,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            pbar.update(size)
    return True


def download_taxi_data(
    datadir: Path,
    start_year: int = 2015,
    end_year: int = 2024,
    delay_between_downloads: float = 1.0,
    max_retries: int = 3,
) -> None:
    """Download all taxi data files and zone metadata to datadir/nyctaxi/orig/.

    Args:
        datadir: Base data directory.
        start_year: First year to download (default 2015).
        end_year: Last year to download (default 2024).
        delay_between_downloads: Seconds to wait between downloads to avoid rate limiting.
        max_retries: Number of retry attempts for failed downloads with exponential backoff.
    """
    orig_dir, _ = get_data_dirs(datadir)

    # Download zone metadata
    zone_path = orig_dir / "taxi_zone_lookup.csv"
    if not zone_path.exists():
        print("Downloading zone metadata...")
        download_file(ZONE_URL, zone_path)

    # Download trip data files
    file_urls = generate_file_urls(start_year, end_year)
    print(f"Downloading {len(file_urls)} taxi data files...")
    failed_downloads = []

    for url, filename in tqdm(file_urls, desc="Overall progress"):
        dest_path = orig_dir / filename
        if dest_path.exists():
            continue

        # Retry with exponential backoff
        for attempt in range(max_retries):
            try:
                download_file(url, dest_path)
                time.sleep(delay_between_downloads)
                break
            except requests.exceptions.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) * 5  # 5s, 10s, 20s
                    print(
                        f"\n{filename} failed (attempt {attempt + 1}), retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    print(
                        f"\nFailed to download {filename} after {max_retries} attempts: {e}"
                    )
                    failed_downloads.append(filename)

    if failed_downloads:
        print(f"\n{len(failed_downloads)} files failed to download:")
        for f in failed_downloads:
            print(f"  - {f}")


def load_zone_mapping(zone_csv_path: Path) -> dict[int, str]:
    """Load zone metadata and return LocationID -> Borough mapping dict."""
    zone_df = pd.read_csv(zone_csv_path)
    return dict(zip(zone_df["LocationID"], zone_df["Borough"]))


def preprocess_file(
    input_path: Path, output_path: Path, zone_mapping: dict[int, str]
) -> None:
    """Add PUBorough and DOBorough columns based on zone mapping, save to output."""
    if output_path.exists():
        return

    df = pd.read_parquet(input_path)

    # Map location IDs to boroughs
    df["PUBorough"] = df["PULocationID"].map(zone_mapping).fillna("Unknown")
    df["DOBorough"] = df["DOLocationID"].map(zone_mapping).fillna("Unknown")

    # Save to output path
    df.to_parquet(output_path, index=False)


def preprocess_all_files(datadir: Path) -> None:
    """Preprocess all files in orig/ to preproc/ with borough columns."""
    orig_dir, preproc_dir = get_data_dirs(datadir)

    # Load zone mapping
    zone_csv_path = orig_dir / "taxi_zone_lookup.csv"
    if not zone_csv_path.exists():
        raise FileNotFoundError(f"Zone metadata not found: {zone_csv_path}")
    zone_mapping = load_zone_mapping(zone_csv_path)

    # Process each file
    parquet_files = sorted(orig_dir.glob("yellow_tripdata_*.parquet"))
    print(f"Preprocessing {len(parquet_files)} files...")
    for input_path in tqdm(parquet_files, desc="Preprocessing"):
        output_path = preproc_dir / input_path.name
        preprocess_file(input_path, output_path, zone_mapping)


def combine_preprocessed_files(datadir: Path) -> Path:
    """Combine all preprocessed files into single parquet, return path."""
    _, preproc_dir = get_data_dirs(datadir)
    combined_path = datadir / "nyctaxi" / "combined_taxi_data.parquet"

    if combined_path.exists():
        print(f"Combined file already exists: {combined_path}")
        return combined_path

    parquet_files = sorted(preproc_dir.glob("yellow_tripdata_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No preprocessed files found in {preproc_dir}")

    print(f"Combining {len(parquet_files)} files...")

    # Use PyArrow for efficient parquet concatenation
    tables = []
    for file_path in tqdm(parquet_files, desc="Reading files"):
        table = pq.read_table(file_path)
        tables.append(table)

    print("Concatenating tables...")
    import pyarrow as pa

    # Standardize column names and unify schemas
    standardized_tables = []
    for table in tables:
        # Rename Airport_fee to airport_fee if present
        if 'Airport_fee' in table.column_names:
            # Rename by reconstructing the table with new column names
            new_names = [name if name != 'Airport_fee' else 'airport_fee' for name in table.column_names]
            table = table.rename_columns(new_names)
        standardized_tables.append(table)
    
    # Build unified schema from all tables, promoting null types to double
    all_fields = {}
    for table in standardized_tables:
        for field in table.schema:
            if field.name not in all_fields:
                all_fields[field.name] = field.type
            elif pa.types.is_null(all_fields[field.name]) and not pa.types.is_null(field.type):
                all_fields[field.name] = field.type
    
    # Promote null types to double for known nullable columns
    for col_name in ['congestion_surcharge', 'airport_fee']:
        if col_name in all_fields and pa.types.is_null(all_fields[col_name]):
            all_fields[col_name] = pa.float64()
    
    # Create unified schema maintaining field order from first table
    unified_schema = pa.schema([pa.field(name, all_fields[name]) for name in standardized_tables[0].schema.names])
    
    # Cast all tables to unified schema
    tables = [table.cast(unified_schema) for table in standardized_tables]

    combined_table = pa.concat_tables(tables)

    print(f"Writing combined file to {combined_path}...")
    pq.write_table(combined_table, combined_path)

    return combined_path


def create_duckdb_database(combined_parquet: Path, db_path: Path) -> None:
    """Create DuckDB database from combined parquet file."""
    if db_path.exists():
        print(f"DuckDB database already exists: {db_path}")
        return

    print(f"Creating DuckDB database at {db_path}...")
    con = duckdb.connect(str(db_path))
    con.execute(f"""
        CREATE TABLE taxi_trips AS
        SELECT * FROM read_parquet('{combined_parquet}')
    """)
    con.close()
    print("DuckDB database created successfully.")


def compute_borough_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PUBorough x DOBorough trip count matrix from dataframe."""
    return pd.crosstab(df["PUBorough"], df["DOBorough"])


def compute_borough_matrix_duckdb(
    con: duckdb.DuckDBPyConnection, year: int, month: int
) -> pd.DataFrame:
    """Compute borough matrix using DuckDB SQL query."""
    query = f"""
        SELECT PUBorough, DOBorough, COUNT(*) as trip_count
        FROM taxi_trips
        WHERE YEAR(tpep_pickup_datetime) = {year}
          AND MONTH(tpep_pickup_datetime) = {month}
        GROUP BY PUBorough, DOBorough
    """
    result = con.execute(query).fetchdf()
    # Pivot to matrix form
    return (
        result.pivot(index="PUBorough", columns="DOBorough", values="trip_count")
        .fillna(0)
        .astype(int)
    )
