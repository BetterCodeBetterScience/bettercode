import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from bettercode.utils import save_fig


@pytest.fixture
def figure():
    """Provide a simple matplotlib figure and clean it up afterward."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    yield fig
    plt.close(fig)


def test_save_fig_writes_to_zero_padded_chapter_dir(figure, tmp_path):
    """The figure is written to basedir/chapternum(zero-padded)/name.pdf."""
    result = save_fig(figure, "scatter", 3, basedir=str(tmp_path))
    expected = tmp_path / "03" / "scatter.pdf"
    assert expected.exists()
    assert str(expected) == str(result)


def test_save_fig_defaults_to_pdf_and_300_dpi(figure, tmp_path):
    """Default file type is PDF."""
    result = save_fig(figure, "lines", 12, basedir=str(tmp_path))
    assert str(result).endswith("12/lines.pdf")
    assert (tmp_path / "12" / "lines.pdf").exists()


def test_save_fig_respects_custom_filetype(figure, tmp_path):
    """A custom file type changes the extension and format."""
    result = save_fig(figure, "bars", 1, filetype="png", basedir=str(tmp_path))
    expected = tmp_path / "01" / "bars.png"
    assert expected.exists()
    assert str(expected) == str(result)


def test_save_fig_creates_missing_directories(figure, tmp_path):
    """Intermediate directories are created if they do not exist."""
    basedir = tmp_path / "figures"
    save_fig(figure, "plot", 7, basedir=str(basedir))
    assert (basedir / "07" / "plot.pdf").exists()
