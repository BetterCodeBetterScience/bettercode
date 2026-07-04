from tex2qmd.preprocess import (
    parse_listing_options,
    render_code_block,
    slice_lines,
    strip_comments,
    unescape_listing_math,
)


def test_strip_comments_removes_full_line_comment():
    """A full-line LaTeX comment outside a listing is removed."""
    tex = "Real prose.\n% a commented line\nMore prose."
    assert strip_comments(tex) == "Real prose.\nMore prose."


def test_strip_comments_removes_commented_command():
    """A commented-out \\lstinputlisting line is removed so it is never processed."""
    tex = "Before.\n% \\lstinputlisting[style=Python]{gone.py}\nAfter."
    assert strip_comments(tex) == "Before.\nAfter."


def test_strip_comments_preserves_percent_inside_listing():
    """A line starting with % inside a lstlisting body is preserved (e.g. %matplotlib)."""
    tex = (
        "\\begin{lstlisting}[style=Pythonshort]\n"
        "%matplotlib inline\n"
        "x = 1\n"
        "\\end{lstlisting}"
    )
    assert "%matplotlib inline" in strip_comments(tex)


def test_strip_comments_keeps_indented_comment_only_when_in_listing():
    """Indented full-line comments in prose are removed; inside listings kept."""
    prose = "text\n   % indented comment\nmore"
    assert strip_comments(prose) == "text\nmore"


def test_slice_lines_inclusive_range():
    """firstline/lastline slice is 1-indexed and inclusive."""
    text = "a\nb\nc\nd\ne\n"
    assert slice_lines(text, 2, 4) == "b\nc\nd"


def test_slice_lines_no_bounds_returns_all():
    """No firstline/lastline returns the full text (trailing newline trimmed)."""
    text = "a\nb\nc\n"
    assert slice_lines(text, None, None) == "a\nb\nc"


def test_slice_lines_firstline_only():
    """firstline with no lastline slices to end."""
    text = "a\nb\nc\nd\n"
    assert slice_lines(text, 3, None) == "c\nd"


def test_parse_listing_options():
    """Option string parses into a key/value dict."""
    opts = "style=Python, firstline=8, lastline=16"
    assert parse_listing_options(opts) == {
        "style": "Python",
        "firstline": "8",
        "lastline": "16",
    }


def test_parse_listing_options_empty():
    """Empty option string yields an empty dict."""
    assert parse_listing_options("") == {}


def test_render_code_block():
    """A fenced block wraps code with the language tag."""
    assert render_code_block("print(1)", "python") == "```python\nprint(1)\n```"


def test_unescape_listing_math_converts_symbols():
    """LaTeX `$...$` math (from a mathescape listing) becomes unicode text."""
    code = "446 $\\mu$s $\\pm$ 1.06 $\\mu$s per loop"
    assert unescape_listing_math(code) == "446 µs ± 1.06 µs per loop"


def test_unescape_listing_math_leaves_plain_dollars():
    """Text without `$...$` math spans is returned unchanged."""
    assert unescape_listing_math("$ git init\n$ echo hello") == "$ git init\n$ echo hello"
