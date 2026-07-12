import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from bettercode.figstyle import (
    HATCHES,
    LINESTYLES,
    MARKERS,
    gray_palette,
    grayscale_rc,
    variant_dirs,
)
from bettercode.utils import save_fig_variants


def test_gray_palette_returns_requested_number_of_colors():
    """gray_palette(n) yields exactly n colors."""
    assert len(gray_palette(1)) == 1
    assert len(gray_palette(4)) == 4
    assert len(gray_palette(7)) == 7


def test_gray_palette_colors_are_achromatic():
    """Every palette entry has equal RGB channels (pure gray)."""
    for color in gray_palette(6):
        r, g, b = to_rgb(color)
        assert r == g == b


def test_gray_palette_is_ordered_dark_to_light():
    """Palette luminance increases monotonically, starting at black."""
    grays = [to_rgb(c)[0] for c in gray_palette(5)]
    assert grays[0] == 0.0
    assert all(a < b for a, b in zip(grays, grays[1:]))


def test_gray_palette_stays_visible_on_white():
    """No palette entry is too light to distinguish from a white background."""
    assert all(to_rgb(c)[0] <= 0.8 for c in gray_palette(8))


def test_style_cycles_provide_redundant_encodings():
    """Linestyle, marker, and hatch cycles each offer at least four distinct entries."""
    for cycle in (LINESTYLES, MARKERS, HATCHES):
        assert len(cycle) >= 4
        assert len(set(cycle)) == len(cycle)
    assert LINESTYLES[0] == "-"


def test_grayscale_rc_sets_greys_colormap_and_gray_prop_cycle():
    """The rc dict maps images to Greys and cycles achromatic line colors."""
    rc = grayscale_rc()
    assert rc["image.cmap"] == "Greys"
    colors = [entry["color"] for entry in rc["axes.prop_cycle"]]
    for color in colors:
        r, g, b = to_rgb(color)
        assert r == g == b


def test_grayscale_rc_is_usable_as_rc_context():
    """Plotting inside rc_context(grayscale_rc()) uses gray line colors."""
    with matplotlib.rc_context(grayscale_rc()):
        fig, ax = plt.subplots()
        (line,) = ax.plot([0, 1], [0, 1])
        r, g, b = to_rgb(line.get_color())
        assert r == g == b
    plt.close(fig)


def test_variant_dirs_maps_figures_dir_to_gray_sibling():
    """variant_dirs inserts a gray/ prefix ahead of the figures directory."""
    dirs = variant_dirs(Path("../../latex/figures"))
    assert dirs["color"] == Path("../../latex/figures")
    assert dirs["gray"] == Path("../../latex/gray/figures")


def test_save_fig_variants_writes_color_and_gray_trees(tmp_path):
    """The factory runs once per variant; color gets pdf+svg, gray gets pdf only."""
    image_dir = tmp_path / "latex" / "figures"
    variants_seen = []

    def make_fig(variant):
        variants_seen.append(variant)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig

    paths = save_fig_variants(make_fig, "demo", 3, image_dir)

    assert variants_seen == ["color", "gray"]
    assert (image_dir / "03" / "demo.pdf").exists()
    assert (image_dir / "03" / "demo.svg").exists()
    gray_dir = tmp_path / "latex" / "gray" / "figures"
    assert (gray_dir / "03" / "demo.pdf").exists()
    assert not (gray_dir / "03" / "demo.svg").exists()
    assert str(paths["color"]).endswith("03/demo.svg") or str(paths["color"]).endswith(
        "03/demo.pdf"
    )
    assert str(paths["gray"]).endswith("03/demo.pdf")


def test_save_fig_variants_closes_figures(tmp_path):
    """No figures remain open after saving both variants."""
    plt.close("all")

    def make_fig(variant):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        return fig

    save_fig_variants(make_fig, "demo", 1, tmp_path / "latex" / "figures")
    assert plt.get_fignums() == []
