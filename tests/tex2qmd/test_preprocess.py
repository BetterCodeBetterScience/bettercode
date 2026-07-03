from tex2qmd.preprocess import slice_lines, parse_listing_options, render_code_block


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
