from tex2qmd.figures import svg_name, rewrite_includegraphics


def test_svg_name():
    """A PDF figure path maps to its .svg sibling."""
    assert svg_name("figures/07/lifecycle.pdf") == "figures/07/lifecycle.svg"


def test_rewrite_includegraphics():
    """includegraphics keeps its LaTeX width; only the path extension changes; pdf collected."""
    tex = "\\includegraphics[width=0.5\\linewidth]{figures/02/LabNotebook.pdf}"
    out, pdfs = rewrite_includegraphics(tex)
    assert out == "\\includegraphics[width=0.5\\linewidth]{figures/02/LabNotebook.svg}"
    assert pdfs == ["figures/02/LabNotebook.pdf"]
