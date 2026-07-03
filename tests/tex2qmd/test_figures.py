from tex2qmd.figures import emit_figure, rewrite_includegraphics, svg_name


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
