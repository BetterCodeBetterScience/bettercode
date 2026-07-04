from tex2qmd.preprocess import extract_listings


def test_extract_inline_listing(tmp_path):
    """Inline lstlisting becomes a token plus a python fenced block."""
    tex = "Before.\n\\begin{lstlisting}[style=Pythonshort]\nprint(1)\n\\end{lstlisting}\nAfter."
    out, blocks = extract_listings(tex, tmp_path)
    assert "TEX2QMDCODEBLOCK0" in out
    assert "\\begin{lstlisting}" not in out
    assert blocks == ["```python\nprint(1)\n```"]


def test_extract_input_listing_with_range(tmp_path):
    """lstinputlisting inlines the referenced file sliced by firstline/lastline."""
    src = tmp_path / "sample.py"
    src.write_text("l1\nl2\nl3\nl4\nl5\n")
    tex = "\\lstinputlisting[style=Python, firstline=2, lastline=4]{sample.py}"
    out, blocks = extract_listings(tex, tmp_path)
    assert out.strip() == "TEX2QMDCODEBLOCK0"
    assert blocks == ["```python\nl2\nl3\nl4\n```"]


def test_extract_mathescape_listing_converts_math(tmp_path):
    """A `mathescape=true` listing has its $...$ math converted to unicode."""
    tex = (
        "\\begin{lstlisting}[style=replshort, mathescape=true]\n"
        "446 $\\mu$s $\\pm$ 1.06 $\\mu$s per loop\n"
        "\\end{lstlisting}"
    )
    _, blocks = extract_listings(tex, tmp_path)
    assert "446 µs ± 1.06 µs per loop" in blocks[0]
    assert "$\\mu$" not in blocks[0]


def test_extract_converts_math_even_without_mathescape_flag(tmp_path):
    """Math symbols convert even when a listing omits the mathescape flag.

    Some source listings (e.g. `[style=repl]`) carry `$\\mu$`/`$\\pm$` output
    without `mathescape=true`; the unambiguous math tokens are still converted.
    """
    tex = (
        "\\begin{lstlisting}[style=repl]\n"
        "103 $\\mu$s $\\pm$ 759 ns per loop\n"
        "\\end{lstlisting}"
    )
    _, blocks = extract_listings(tex, tmp_path)
    assert "103 µs ± 759 ns per loop" in blocks[0]


def test_extract_non_mathescape_listing_preserves_dollar(tmp_path):
    """A literal $ shell prompt is never mistaken for a math delimiter."""
    tex = "\\begin{lstlisting}[style=shellshort]\n$ git init\n\\end{lstlisting}"
    _, blocks = extract_listings(tex, tmp_path)
    assert "$ git init" in blocks[0]


def test_extract_multiple_listings_indexed(tmp_path):
    """Multiple listings get sequential tokens."""
    tex = (
        "\\begin{lstlisting}[style=shellshort]\n$ ls\n\\end{lstlisting}\n"
        "\\begin{lstlisting}[style=Pythonshort]\nx = 1\n\\end{lstlisting}"
    )
    out, blocks = extract_listings(tex, tmp_path)
    assert "TEX2QMDCODEBLOCK0" in out and "TEX2QMDCODEBLOCK1" in out
    assert blocks[0] == "```bash\n$ ls\n```"
    assert blocks[1] == "```python\nx = 1\n```"
