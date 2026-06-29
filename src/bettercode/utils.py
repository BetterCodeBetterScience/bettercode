from pathlib import Path

from matplotlib.figure import Figure


def save_fig(
    fig: Figure,
    name: str,
    chapter: int,
    filetype: str = "pdf",
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
    outpath = outdir / f"{name}.{filetype}"
    fig.savefig(outpath.as_posix(), format=filetype, dpi=dpi, bbox_inches="tight")
    return str(outpath)
