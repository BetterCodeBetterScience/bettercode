import shutil

import pytest
from tex2qmd.figures import (
    MAX_SVG_BYTES,
    convert_pdf_to_png,
    emit_figure,
    png_name,
    rewrite_includegraphics,
    svg_name,
)


@pytest.fixture
def figure_src(tmp_path):
    """A source dir holding figures/01/x.pdf (fake contents)."""
    src = tmp_path / "src"
    (src / "figures" / "01").mkdir(parents=True)
    (src / "figures" / "01" / "x.pdf").write_text("%PDF-fake")
    return src


@pytest.fixture
def png_calls(monkeypatch):
    """Record convert_pdf_to_png calls, writing a stub PNG for each."""
    calls = []

    def _fake(pdf, png):
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(b"PNG")
        calls.append((pdf, png))

    monkeypatch.setattr("tex2qmd.figures.convert_pdf_to_png", _fake)
    return calls


def _write_sized(path, n_bytes):
    """Write a (sparse) file of exactly n_bytes."""
    with open(path, "wb") as f:
        f.truncate(n_bytes)


def test_svg_name():
    """A PDF figure path maps to its .svg sibling."""
    assert svg_name("figures/07/lifecycle.pdf") == "figures/07/lifecycle.svg"


def test_emit_figure_prefers_native_svg(tmp_path, monkeypatch):
    """A native .svg beside the .pdf is copied verbatim; no conversion runs."""
    src = tmp_path / "src"
    (src / "figures" / "01").mkdir(parents=True)
    (src / "figures" / "01" / "x.pdf").write_text("%PDF-fake")
    (src / "figures" / "01" / "x.svg").write_text("<svg>native</svg>")
    out = tmp_path / "out"
    calls = []
    monkeypatch.setattr("tex2qmd.figures.convert_pdf_to_svg", lambda *a, **k: calls.append(a))
    method = emit_figure("figures/01/x.pdf", src, out)
    assert method == "copied"
    assert (out / "figures" / "01" / "x.svg").read_text() == "<svg>native</svg>"
    assert calls == []


def test_emit_figure_converts_when_no_native_svg(tmp_path, monkeypatch):
    """Without a native .svg, the .pdf is converted via pdftocairo."""
    src = tmp_path / "src"
    (src / "figures" / "01").mkdir(parents=True)
    (src / "figures" / "01" / "x.pdf").write_text("%PDF-fake")
    out = tmp_path / "out"
    calls = []
    monkeypatch.setattr(
        "tex2qmd.figures.convert_pdf_to_svg", lambda pdf, svg: calls.append((pdf, svg))
    )
    method = emit_figure("figures/01/x.pdf", src, out)
    assert method == "converted"
    assert calls == [(src / "figures/01/x.pdf", out / "figures/01/x.svg")]


def test_rewrite_includegraphics():
    """includegraphics keeps its LaTeX width; only the path extension changes; pdf collected."""
    tex = "\\includegraphics[width=0.5\\linewidth]{figures/02/LabNotebook.pdf}"
    out, pdfs = rewrite_includegraphics(tex)
    assert out == "\\includegraphics[width=0.5\\linewidth]{figures/02/LabNotebook.svg}"
    assert pdfs == ["figures/02/LabNotebook.pdf"]


def test_png_name():
    """A PDF figure path maps to its .png sibling."""
    assert png_name("figures/07/lifecycle.pdf") == "figures/07/lifecycle.png"


def test_max_svg_bytes_is_ten_megabytes():
    """SVGs above 10 MB are considered too large for the web edition."""
    assert MAX_SVG_BYTES == 10 * 1024 * 1024


def test_emit_figure_rasterizes_oversized_native_svg(figure_src, tmp_path, png_calls):
    """A native .svg above the size limit is replaced by a PNG rendered from the .pdf."""
    _write_sized(figure_src / "figures" / "01" / "x.svg", MAX_SVG_BYTES + 1)
    out = tmp_path / "out"
    method = emit_figure("figures/01/x.pdf", figure_src, out)
    assert method == "rasterized"
    assert not (out / "figures" / "01" / "x.svg").exists()
    assert png_calls == [
        (figure_src / "figures/01/x.pdf", out / "figures/01/x.png")
    ]


def test_emit_figure_keeps_svg_at_size_limit(figure_src, tmp_path, png_calls):
    """An SVG exactly at the size limit is kept as SVG."""
    _write_sized(figure_src / "figures" / "01" / "x.svg", MAX_SVG_BYTES)
    out = tmp_path / "out"
    method = emit_figure("figures/01/x.pdf", figure_src, out)
    assert method == "copied"
    assert (out / "figures" / "01" / "x.svg").exists()
    assert png_calls == []


def test_emit_figure_rasterizes_oversized_converted_svg(
    figure_src, tmp_path, monkeypatch, png_calls
):
    """An SVG converted from the .pdf that exceeds the limit is also rasterized."""
    monkeypatch.setattr(
        "tex2qmd.figures.convert_pdf_to_svg",
        lambda pdf, svg: _write_sized(svg, MAX_SVG_BYTES + 1),
    )
    out = tmp_path / "out"
    (out / "figures" / "01").mkdir(parents=True)
    method = emit_figure("figures/01/x.pdf", figure_src, out)
    assert method == "rasterized"
    assert not (out / "figures" / "01" / "x.svg").exists()
    assert (out / "figures" / "01" / "x.png").exists()


def test_emit_figure_honors_custom_size_limit(figure_src, tmp_path, png_calls):
    """The size limit can be overridden per call."""
    (figure_src / "figures" / "01" / "x.svg").write_text("<svg>native</svg>")
    out = tmp_path / "out"
    method = emit_figure("figures/01/x.pdf", figure_src, out, max_svg_bytes=5)
    assert method == "rasterized"


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("pdftocairo") is None, reason="pdftocairo not installed")
def test_convert_pdf_to_png_renders_png(tmp_path):
    """pdftocairo renders a single-page PDF to a PNG at the requested path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdf = tmp_path / "fig.pdf"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(pdf)
    plt.close(fig)
    out_png = tmp_path / "out" / "figures" / "fig.png"
    convert_pdf_to_png(pdf, out_png)
    assert out_png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
