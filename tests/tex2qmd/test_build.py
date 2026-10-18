import shutil
import subprocess
import sys as _sys
from pathlib import Path

import pytest
from tex2qmd.convert import build_book

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_build_book_writes_project(tmp_path):
    """build_book emits a cover index.qmd and chapter qmd, excludes the preface, and copies bib."""
    latex = tmp_path / "latex"
    latex.mkdir()
    (latex / "book.tex").write_text(
        "\\begin{document}\n\\include{preface}\n\\include{chap1}\n\\end{document}\n"
    )
    (latex / "preface.tex").write_text("\\chapter*{Preface}\nHello.\n")
    (latex / "chap1.tex").write_text("\\chapter{One}\nBody \\citep{k}.\n")
    (latex / "references.bib").write_text("@book{k, title={T}, year={2020}}\n")
    out = tmp_path / "web"
    build_book(latex, out)
    assert (out / "_quarto.yml").exists()
    # index.qmd is a generated cover page, not the preface
    index = (out / "index.qmd").read_text()
    # cover is unnumbered (no front-matter title) so it isn't counted as a chapter
    assert "\ntitle:" not in index
    assert "{.unnumbered}" in index
    assert "Welcome to the web edition" in index
    assert "Hello." not in index
    # the preface is excluded from the web edition entirely
    assert not (out / "preface.qmd").exists()
    assert (out / "chap1.qmd").exists()
    assert (out / "references.bib").exists()
    assert (out / "cambridge.csl").exists()
    yml = (out / "_quarto.yml").read_text()
    assert "type: book" in yml
    # the book title now lives in the config (not the cover front matter)
    assert 'title: "Better Code, Better Science"' in yml
    assert "preface.qmd" not in yml
    # cover is first, chapter follows
    assert yml.index("index.qmd") < yml.index("chap1.qmd")


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_convert_cli_runs_as_script(tmp_path):
    """Running convert.py as a script builds the project (sys.path bootstrap works)."""
    latex = tmp_path / "latex"
    latex.mkdir()
    (latex / "book.tex").write_text(
        "\\begin{document}\n\\include{preface}\n\\include{chap1}\n\\end{document}\n"
    )
    (latex / "preface.tex").write_text("\\chapter*{Preface}\nHello.\n")
    (latex / "chap1.tex").write_text("\\chapter{One}\nBody.\n")
    (latex / "references.bib").write_text("@book{k, title={T}, year={2020}}\n")
    out = tmp_path / "web"
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "tex2qmd" / "convert.py"
    result = subprocess.run(
        [_sys.executable, str(script), "--latex-dir", str(latex), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "_quarto.yml").exists()
    assert (out / "index.qmd").exists()


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_build_book_references_png_for_oversized_svg(tmp_path, monkeypatch):
    """A figure whose SVG exceeds the size limit is referenced as .png in the qmd."""
    from tex2qmd.figures import MAX_SVG_BYTES

    latex = tmp_path / "latex"
    (latex / "figures" / "01").mkdir(parents=True)
    (latex / "book.tex").write_text(
        "\\begin{document}\n\\include{chap1}\n\\end{document}\n"
    )
    (latex / "chap1.tex").write_text(
        "\\chapter{One}\n"
        "\\begin{figure}\n\\includegraphics{figures/01/big.pdf}\n"
        "\\caption{Big.}\n\\label{big-fig}\n\\end{figure}\n"
        "\\begin{figure}\n\\includegraphics{figures/01/small.pdf}\n"
        "\\caption{Small.}\n\\label{small-fig}\n\\end{figure}\n"
    )
    (latex / "references.bib").write_text("@book{k, title={T}, year={2020}}\n")
    for name in ("big", "small"):
        (latex / "figures" / "01" / f"{name}.pdf").write_text("%PDF-fake")
    with open(latex / "figures" / "01" / "big.svg", "wb") as f:
        f.truncate(MAX_SVG_BYTES + 1)
    (latex / "figures" / "01" / "small.svg").write_text("<svg/>")
    monkeypatch.setattr(
        "tex2qmd.figures.convert_pdf_to_png", lambda pdf, png: png.write_bytes(b"PNG")
    )
    out = tmp_path / "web"
    build_book(latex, out)
    qmd = (out / "chap1.qmd").read_text()
    assert "(figures/01/big.png)" in qmd
    assert "big.svg" not in qmd
    assert "(figures/01/small.svg)" in qmd
    assert (out / "figures" / "01" / "big.png").exists()
