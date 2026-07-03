import shutil
import pytest
from tex2qmd.pandoc_bridge import run_pandoc

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_run_pandoc_converts_section_and_citation():
    """Pandoc converts a LaTeX section and \\citep to markdown."""
    latex = "\\section{Intro}\nSee \\citep{smith2020} for details."
    md = run_pandoc(latex)
    assert "# Intro" in md
    assert "[@smith2020]" in md


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_run_pandoc_preserves_placeholder_token():
    """A code placeholder token survives pandoc as plain text."""
    latex = "Before.\n\nTEX2QMDCODEBLOCK0\n\nAfter."
    md = run_pandoc(latex)
    assert "TEX2QMDCODEBLOCK0" in md
