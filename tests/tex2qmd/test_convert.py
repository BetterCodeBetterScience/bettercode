import shutil
import pytest
from tex2qmd.convert import convert_chapter

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_convert_chapter_end_to_end(tmp_path):
    """A small chapter converts: prose to md, listing to fence, index gone, fig collected."""
    (tmp_path / "sample.py").write_text("print('hi')\n")
    tex = (
        "\\section{Demo}\\index{demo}\n"
        "Text with \\citep{ref1}.\n"
        "\\includegraphics[width=0.5\\linewidth]{figures/01/x.pdf}\n"
        "\\lstinputlisting[style=Python]{sample.py}\n"
    )
    qmd, pdfs = convert_chapter(tex, tmp_path)
    assert "# Demo" in qmd
    assert "[@ref1]" in qmd
    assert "\\index" not in qmd
    assert "```python\nprint('hi')\n```" in qmd
    assert pdfs == ["figures/01/x.pdf"]
    assert "figures/01/x.svg" in qmd


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_convert_chapter_renders_input_fragment_box(tmp_path):
    """An \\input'd fullbox becomes a titled callout, with its model quote intact."""
    (tmp_path / "fragments").mkdir()
    (tmp_path / "fragments" / "box.tex").write_text(
        "\\begin{fullbox}[\\textit{GPT-4}'s coding hurdle]\n"
        "Box prose here.\n"
        "\\begin{modelquote}\n"
        "The model said this.\n"
        "\\end{modelquote}\n"
        "\\end{fullbox}\n"
    )
    tex = "\\section{Demo}\n\\input{fragments/box}\n"
    qmd, _ = convert_chapter(tex, tmp_path)
    assert "{.callout-note}" in qmd
    assert "## *GPT-4*'s coding hurdle" in qmd
    assert "Box prose here." in qmd
    assert "> The model said this." in qmd
    assert "fullbox" not in qmd
    assert "modelquote" not in qmd


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_convert_chapter_keeps_noindent_prose(tmp_path):
    """Prose wrapped in `\\noindent{...}` survives conversion (pandoc alone drops it)."""
    tex = "\\noindent{The \\textit{hash} uniquely names the commit.}\n"
    qmd, _ = convert_chapter(tex, tmp_path)
    assert "The *hash* uniquely names the commit." in qmd
