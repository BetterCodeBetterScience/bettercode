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
