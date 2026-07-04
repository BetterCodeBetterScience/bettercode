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
    assert 'title: "Better Code, Better Science"' in index
    assert "Welcome to the web edition" in index
    assert "Hello." not in index
    # the preface is excluded from the web edition entirely
    assert not (out / "preface.qmd").exists()
    assert (out / "chap1.qmd").exists()
    assert (out / "references.bib").exists()
    assert (out / "cambridge.csl").exists()
    yml = (out / "_quarto.yml").read_text()
    assert "type: book" in yml
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
