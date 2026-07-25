from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from bettercode.figstyle import variant_dirs


def save_fig(
    fig: Figure,
    name: str,
    chapter: int,
    filetypes: list[str] | None = None,
    dpi: int = 300,
    basedir: str = "figures",
) -> str:
    """Save a matplotlib figure to basedir/<zero-padded chapter>/<name>.<filetype>.

    Returns the path to the saved file.
    """
    # if basedir is not a Path object, convert it
    if basedir is not Path:
        basedir = Path(basedir)
    outdir = basedir / f"{chapter:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    if filetypes is None:
        filetypes = ["pdf", "svg"]
    for filetype in filetypes:
        outpath = outdir / f"{name}.{filetype}"
        fig.savefig(outpath.as_posix(), format=filetype, dpi=dpi, bbox_inches="tight")
    return str(outpath)


def save_fig_variants(
    make_fig: Callable[[str], Figure],
    name: str,
    chapter: int,
    image_dir: str | Path,
    dpi: int = 300,
) -> dict[str, str]:
    """Render and save color and grayscale variants of a figure.

    Calls make_fig("color") and make_fig("gray") and saves each into the
    matching tree from variant_dirs; color as pdf+svg, gray as pdf only.
    Returns the saved path per variant.
    """
    dirs = variant_dirs(Path(image_dir))
    filetypes = {"color": ["pdf", "svg"], "gray": ["pdf"]}
    paths = {}
    for variant in ("color", "gray"):
        fig = make_fig(variant)
        paths[variant] = save_fig(
            fig,
            name,
            chapter,
            filetypes=filetypes[variant],
            dpi=dpi,
            basedir=dirs[variant],
        )
        plt.close(fig)
    return paths


def get_size(path: str | Path, units: str = "M") -> float:
    """Total size of a file, or a directory's contents (recursive), in the given units.

    Args:
        units: One of "B", "K", "M", "G" for bytes, kibi-, mebi-, or gibibytes.
    """
    factors = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    path = Path(path)
    if path.is_dir():
        nbytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        nbytes = path.stat().st_size
    return nbytes / factors[units]
