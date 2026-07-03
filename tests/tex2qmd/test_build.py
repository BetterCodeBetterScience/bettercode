import shutil
import pytest
from tex2qmd.convert import build_book

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_build_book_writes_project(tmp_path):
    """build_book emits _quarto.yml, index.qmd, chapter qmd, and copies bib."""
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
    assert (out / "index.qmd").exists()
    assert (out / "chap1.qmd").exists()
    assert (out / "references.bib").exists()
    assert (out / "cambridge.csl").exists()
    assert "type: book" in (out / "_quarto.yml").read_text()
