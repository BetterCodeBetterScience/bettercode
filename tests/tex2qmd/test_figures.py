from tex2qmd.figures import latex_width_to_percent, svg_name, rewrite_includegraphics


def test_latex_width_to_percent():
    """A fractional linewidth becomes an integer percentage."""
    assert latex_width_to_percent("0.625\\linewidth") == "62%"
    assert latex_width_to_percent("0.5\\textwidth") == "50%"


def test_latex_width_to_percent_unparseable():
    """A width with no recognizable fraction returns None."""
    assert latex_width_to_percent("3cm") is None


def test_svg_name():
    """A PDF figure path maps to its .svg sibling."""
    assert svg_name("figures/07/lifecycle.pdf") == "figures/07/lifecycle.svg"


def test_rewrite_includegraphics():
    """includegraphics is rewritten to svg with a width attribute; pdf collected."""
    tex = "\\includegraphics[width=0.5\\linewidth]{figures/02/LabNotebook.pdf}"
    out, pdfs = rewrite_includegraphics(tex)
    assert out == "\\includegraphics[width=50%]{figures/02/LabNotebook.svg}"
    assert pdfs == ["figures/02/LabNotebook.pdf"]
