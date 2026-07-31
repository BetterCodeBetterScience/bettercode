import pytest

from tex2qmd.preprocess import (
    mark_box_titles,
    parse_listing_options,
    render_code_block,
    resolve_inputs,
    slice_lines,
    strip_comments,
    unescape_listing_math,
    unwrap_noindent,
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


def test_resolve_inputs_inlines_fragment(tmp_path):
    """An \\input file's content replaces the command, so pandoc never sees it."""
    (tmp_path / "fragments").mkdir()
    (tmp_path / "fragments" / "box.tex").write_text("Boxed prose.\n")
    tex = "Before.\n\\input{fragments/box}\nAfter."
    assert resolve_inputs(tex, tmp_path) == "Before.\nBoxed prose.\n\nAfter."


def test_resolve_inputs_accepts_explicit_tex_suffix(tmp_path):
    """An \\input target that already ends in .tex is not given a second suffix."""
    (tmp_path / "box.tex").write_text("Boxed.")
    assert resolve_inputs("\\input{box.tex}", tmp_path) == "Boxed."


def test_resolve_inputs_is_recursive(tmp_path):
    """An \\input inside an included file is resolved too."""
    (tmp_path / "outer.tex").write_text("outer\n\\input{inner}")
    (tmp_path / "inner.tex").write_text("inner")
    assert resolve_inputs("\\input{outer}", tmp_path) == "outer\ninner"


def test_resolve_inputs_ignores_trailing_whitespace(tmp_path):
    """Trailing spaces after \\input{...} (as in the chapters) do not break resolution."""
    (tmp_path / "box.tex").write_text("Boxed.")
    assert resolve_inputs("\\input{box}    \n", tmp_path) == "Boxed.    \n"


def test_resolve_inputs_leaves_text_without_input(tmp_path):
    """Text with no \\input command is returned unchanged."""
    assert resolve_inputs("Just prose.", tmp_path) == "Just prose."


def test_resolve_inputs_missing_file_raises(tmp_path):
    """A missing \\input target is an error, not silently dropped content."""
    with pytest.raises(FileNotFoundError):
        resolve_inputs("\\input{nope}", tmp_path)


def test_unwrap_noindent_unwraps_braced_group():
    """`\\noindent{...}` becomes bare text; pandoc would otherwise swallow the group."""
    assert unwrap_noindent("\\noindent{Note that the symbol matters.}") == (
        "Note that the symbol matters."
    )


def test_unwrap_noindent_handles_nested_braces():
    """Brace matching (not the first `}`) delimits the group being unwrapped."""
    tex = "\\noindent{The \\textit{hash} is {\\textasciitilde}8 chars.}"
    assert unwrap_noindent(tex) == "The \\textit{hash} is {\\textasciitilde}8 chars."


def test_unwrap_noindent_handles_multiple_occurrences():
    """Every `\\noindent{...}` in the text is unwrapped."""
    tex = "\\noindent{One.}\n\n\\noindent{Two.}"
    assert unwrap_noindent(tex) == "One.\n\nTwo."


def test_unwrap_noindent_drops_bare_command():
    """A bare `\\noindent` with no group is removed and its text left in place."""
    assert unwrap_noindent("\\bigskip\\noindent\nFollowing text.") == (
        "\\bigskip\nFollowing text."
    )


def test_unwrap_noindent_ignores_escaped_brace():
    """An escaped `\\{` inside the group does not confuse brace matching."""
    tex = "\\noindent{A \\{literal\\} brace.}"
    assert unwrap_noindent(tex) == "A \\{literal\\} brace."


def test_mark_box_titles_moves_optional_title_into_body():
    """A fullbox's `[title]` becomes a tokenized paragraph pandoc will render."""
    tex = "\\begin{fullbox}[\\textit{GPT-4}'s hurdle]\nBody.\n\\end{fullbox}"
    out = mark_box_titles(tex, 5)
    assert "\\begin{fullbox}\n" in out
    assert "TEX2QMDBOXTITLE{Box 5.1 --- \\textit{GPT-4}'s hurdle}" in out
    assert "[\\textit{GPT-4}'s hurdle]" not in out
    assert "Body." in out


def test_mark_box_titles_separates_token_from_following_prose():
    """A blank line follows the token so it is always its own block."""
    tex = "\\begin{fullbox}[Title]\nBody.\n\\end{fullbox}"
    assert "TEX2QMDBOXTITLE{Box 3.1 --- Title}\n\nBody." in mark_box_titles(tex, 3)


def test_mark_box_titles_numbers_boxes_within_the_chapter():
    """Box numbers count up within a chapter, matching the LaTeX counter."""
    tex = (
        "\\begin{fullbox}[First]\na\n\\end{fullbox}\n"
        "\\begin{fullbox}[Second]\nb\n\\end{fullbox}\n"
    )
    out = mark_box_titles(tex, 2)
    assert "{Box 2.1 --- First}" in out
    assert "{Box 2.2 --- Second}" in out


def test_mark_box_titles_numbers_untitled_box():
    """A fullbox with no optional title still gets its number as the title."""
    tex = "\\begin{fullbox}\nBody.\n\\end{fullbox}"
    out = mark_box_titles(tex, 4)
    assert "TEX2QMDBOXTITLE{Box 4.1}" in out


def test_mark_box_titles_leaves_other_environments_untouched():
    """Only fullbox openings are rewritten."""
    tex = "\\begin{figure}[htbp]\nstuff\n\\end{figure}"
    assert mark_box_titles(tex, 1) == tex
