"""Styling helpers for rendering grayscale variants of book figures.

The print edition requires grayscale figures, so multi-series plots need
redundant encodings (linestyle, marker, hatch) in addition to gray levels.
"""

from pathlib import Path

from cycler import cycler

LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1)))
MARKERS = ("o", "s", "^", "D", "v", "X")
HATCHES = ("", "///", "...", "xxx", "\\\\\\", "++")

_MAX_GRAY = 0.75


def gray_palette(n: int) -> list[str]:
    """n achromatic colors ordered dark to light, starting at black."""
    if n == 1:
        return ["0.00"]
    return [f"{_MAX_GRAY * i / (n - 1):.2f}" for i in range(n)]


def grayscale_rc() -> dict:
    """Matplotlib rc overrides for grayscale rendering, for use with rc_context."""
    n = min(len(LINESTYLES), 4)
    return {
        "image.cmap": "Greys",
        "axes.prop_cycle": cycler(color=gray_palette(n))
        + cycler(linestyle=list(LINESTYLES[:n])),
    }


def variant_dirs(image_dir: Path) -> dict[str, Path]:
    """Map a figure directory to its color and grayscale output directories.

    The gray tree mirrors the color tree under a gray/ prefix, e.g.
    latex/figures -> latex/gray/figures.
    """
    image_dir = Path(image_dir)
    return {
        "color": image_dir,
        "gray": image_dir.parent / "gray" / image_dir.name,
    }
