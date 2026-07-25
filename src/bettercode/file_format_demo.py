import pandas as pd
import numpy as np
import os
import templateflow.api as tf
import nibabel as nib
from pathlib import Path
import time
import h5py
import hdf5plugin
import zarr
from zarr.codecs import GzipCodec, BloscCodec
from bettercode.utils import save_fig, get_size
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

os.environ["TEMPLATEFLOW_HOME"] = "/Users/poldrack/.cache/templateflow"
IMAGE_DIR = Path("../../latex/figures")
CHAPTER = 7

### Multidimensional arrays

difumo = tf.get(
    "MNI152NLin6Asym",
    atlas="DiFuMo",
    desc="512dimensions",
    resolution=2,
    suffix="probseg",
    extension=".nii.gz",
)
img = nib.load(difumo)
difumo_data = img.get_fdata()

# matched gzip level and Blosc settings so h5 and zarr are compared fairly
GZIP_LEVEL = 4
H5_BLOSC = hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=1)
ZARR_BLOSC = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")


def write_h5(path, data, **kwargs):
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, **kwargs)


def write_zarr(path, data, compressors):
    zarr.create_array(path, data=data, compressors=compressors, overwrite=True)


def read_h5(path):
    with h5py.File(path, "r") as f:
        return f["data"][:]


def read_zarr(path):
    return zarr.open(path, mode="r")[:]


# each row: (format, compression, path, writer, reader)
storage_configs = [
    ("npy", "none", "/tmp/difumo.npy", lambda p, d: np.save(p, d), np.load),
    (
        "h5",
        "gzip",
        "/tmp/difumo_gzip.h5",
        lambda p, d: write_h5(p, d, compression="gzip", compression_opts=GZIP_LEVEL),
        read_h5,
    ),
    (
        "h5",
        "blosc",
        "/tmp/difumo_blosc.h5",
        lambda p, d: write_h5(p, d, **H5_BLOSC),
        read_h5,
    ),
    (
        "zarr",
        "gzip",
        "/tmp/difumo_gzip.zarr",
        lambda p, d: write_zarr(p, d, GzipCodec(level=GZIP_LEVEL)),
        read_zarr,
    ),
    (
        "zarr",
        "blosc",
        "/tmp/difumo_blosc.zarr",
        lambda p, d: write_zarr(p, d, ZARR_BLOSC),
        read_zarr,
    ),
]


def _remove(path):
    """Remove a file, or a directory such as a zarr store."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def benchmark_storage(data, configs, nreps=5):
    """Time writing and reading each format nreps times; one row per repetition."""
    records = []
    for fmt, comp, path, writer, reader in configs:
        print(f"Benchmarking {fmt} with {comp} compression...")
        for _ in range(nreps):
            # start from a clean slate so deletion isn't counted in the write time
            _remove(path)

            start = time.time()
            writer(path, data)
            write_time = time.time() - start

            start = time.time()
            _ = reader(path)
            read_time = time.time() - start

            records.append({
                "format": fmt,
                "compression": comp,
                "size_mb": get_size(path),
                "write_s": write_time,
                "read_s": read_time,
            })
    return pd.DataFrame(records)


results_df = benchmark_storage(difumo_data, storage_configs)
print(
    results_df.groupby(["format", "compression"]).mean(numeric_only=True).to_markdown()
)


# visualize file size and read/write times across the format/compression matrix.
# with one row per repetition, seaborn draws the mean and its variability (sd) for us.


sns.set_style("whitegrid")

metrics = [
    ("size_mb", "File size (MB)"),
    ("write_s", "Write time (s)"),
    ("read_s", "Read time (s)"),
]
sns.set_theme(style="whitegrid", rc={"xtick.labelsize": 12})
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (col, title) in zip(axes, metrics):
    sns.barplot(
        data=results_df, x="format", y=col, hue="compression", ax=ax, errorbar="sd"
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel(title, fontsize=12)
# keep a single legend rather than one per panel
for ax in axes[:-1]:
    ax.get_legend().remove()

fig.tight_layout()
save_fig(fig, "storage_benchmark", CHAPTER, basedir=IMAGE_DIR)
