"""Tests for the gray-figure gap-fill tool (scripts/build_gray_figures.py)."""

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_gray_figures import (
    convert_pdf_to_gray,
    convert_png_to_gray,
    fill_gray_tree,
    find_figure_refs,
    verify_complete,
)

needs_gs = pytest.mark.skipif(
    shutil.which("gs") is None, reason="ghostscript not installed"
)


@pytest.fixture
def latex_dir(tmp_path):
    """A miniature latex tree: one chapter .tex, one color PDF, one color PNG."""
    (tmp_path / "figures" / "01").mkdir(parents=True)
    (tmp_path / "files").mkdir()

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="red", linewidth=4)
    ax.set_facecolor("lightblue")
    fig.savefig(tmp_path / "figures" / "01" / "colorplot.pdf")
    plt.close(fig)

    Image.new("RGB", (32, 32), color=(200, 30, 30)).save(
        tmp_path / "files" / "photo.png"
    )

    (tmp_path / "book-demo.tex").write_text(
        "Some prose.\n"
        "\\includegraphics[width=0.75\\linewidth]{figures/01/colorplot.pdf}\n"
        "More prose.\n"
        "% \\includegraphics{figures/01/commented_out.pdf}\n"
        "\\includegraphics{files/photo.png}\n"
    )
    return tmp_path


def test_find_figure_refs_parses_paths_and_skips_comments(latex_dir):
    """Both include forms are found; commented-out includes are ignored."""
    refs = find_figure_refs(latex_dir)
    assert Path("figures/01/colorplot.pdf") in refs
    assert Path("files/photo.png") in refs
    assert Path("figures/01/commented_out.pdf") not in refs


@needs_gs
def test_convert_pdf_to_gray_removes_all_color(latex_dir, tmp_path):
    """The converted PDF has zero cyan/magenta/yellow ink coverage."""
    src = latex_dir / "figures" / "01" / "colorplot.pdf"
    dest = tmp_path / "gray.pdf"
    convert_pdf_to_gray(src, dest)
    assert dest.exists()
    out = subprocess.run(
        ["gs", "-o", "-", "-sDEVICE=inkcov", str(dest)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    coverage = [line.split() for line in out.splitlines() if "CMYK OK" in line]
    assert coverage, f"no inkcov output in: {out}"
    for c, m, y, _k in (vals[:4] for vals in coverage):
        assert float(c) == float(m) == float(y) == 0.0


def test_convert_png_to_gray_produces_achromatic_image(latex_dir, tmp_path):
    """Every pixel of the converted PNG has equal RGB channels."""
    dest = tmp_path / "gray.png"
    convert_png_to_gray(latex_dir / "files" / "photo.png", dest)
    pixels = np.asarray(Image.open(dest).convert("RGB"))
    assert (pixels[..., 0] == pixels[..., 1]).all()
    assert (pixels[..., 1] == pixels[..., 2]).all()


@needs_gs
def test_fill_gray_tree_converts_missing_and_reports(latex_dir):
    """Referenced figures without a gray variant are converted into gray/."""
    report = fill_gray_tree(latex_dir)
    assert (latex_dir / "gray" / "figures" / "01" / "colorplot.pdf").exists()
    assert (latex_dir / "gray" / "files" / "photo.png").exists()
    assert sorted(report.converted) == [
        Path("figures/01/colorplot.pdf"),
        Path("files/photo.png"),
    ]
    assert report.missing_sources == []


@needs_gs
def test_fill_gray_tree_preserves_existing_gray_variants(latex_dir):
    """A gray variant already produced by a notebook is not overwritten."""
    gray_pdf = latex_dir / "gray" / "figures" / "01" / "colorplot.pdf"
    gray_pdf.parent.mkdir(parents=True)
    gray_pdf.write_bytes(b"notebook-made gray variant")
    report = fill_gray_tree(latex_dir)
    assert gray_pdf.read_bytes() == b"notebook-made gray variant"
    assert Path("figures/01/colorplot.pdf") not in report.converted
    assert Path("figures/01/colorplot.pdf") in report.skipped


def test_fill_gray_tree_records_missing_sources(latex_dir):
    """A reference with no color source file is reported, not fatal."""
    (latex_dir / "book-demo.tex").write_text(
        "\\includegraphics{figures/01/nonexistent.pdf}\n"
    )
    report = fill_gray_tree(latex_dir)
    assert Path("figures/01/nonexistent.pdf") in report.missing_sources


@needs_gs
def test_verify_complete_reports_missing_gray_variants(latex_dir):
    """verify_complete lists referenced figures absent from the gray tree."""
    missing_before = verify_complete(latex_dir)
    assert sorted(missing_before) == [
        Path("figures/01/colorplot.pdf"),
        Path("files/photo.png"),
    ]
    fill_gray_tree(latex_dir)
    assert verify_complete(latex_dir) == []
