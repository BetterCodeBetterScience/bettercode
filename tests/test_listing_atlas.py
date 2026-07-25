"""Tests for the listing-atlas tooling (scripts/listing_atlas.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from listing_atlas import best_source, coverage, extract_listings, match_listing


# --- extract_listings -------------------------------------------------------


def test_extract_single_listing_captures_style_and_code():
    """A single styled block yields its style and verbatim code lines."""
    tex = (
        "Some prose.\n"
        "\\begin{lstlisting}[style=Python]\n"
        "def f(x):\n"
        "    return x + 1\n"
        "\\end{lstlisting}\n"
        "More prose.\n"
    )
    listings = extract_listings(tex)
    assert len(listings) == 1
    assert listings[0].style == "Python"
    assert listings[0].code == ["def f(x):", "    return x + 1"]


def test_extract_records_one_based_tex_line_numbers():
    """begin_line/end_line are 1-based line numbers of the delimiters."""
    tex = (
        "line1\n"
        "\\begin{lstlisting}[style=Python]\n"  # line 2
        "code\n"  # line 3
        "\\end{lstlisting}\n"  # line 4
    )
    listing = extract_listings(tex)[0]
    assert listing.begin_line == 2
    assert listing.end_line == 4


def test_extract_listing_without_options_has_none_style():
    """A bare \\begin{lstlisting} (no [options]) records style None."""
    tex = "\\begin{lstlisting}\nhello\n\\end{lstlisting}\n"
    assert extract_listings(tex)[0].style is None


def test_extract_returns_multiple_blocks_in_order():
    """Multiple blocks are returned in document order."""
    tex = (
        "\\begin{lstlisting}[style=Python]\na\n\\end{lstlisting}\n"
        "prose\n"
        "\\begin{lstlisting}[style=shell]\nb\n\\end{lstlisting}\n"
    )
    listings = extract_listings(tex)
    assert [item.style for item in listings] == ["Python", "shell"]


# --- match_listing ----------------------------------------------------------


def test_match_finds_contiguous_block_line_range():
    """An exact contiguous match reports 1-based source line range and score 1.0."""
    source = [
        "import os",  # 1
        "",  # 2
        "def f(x):",  # 3
        "    return x + 1",  # 4
        "",  # 5
        "print('done')",  # 6
    ]
    code = ["def f(x):", "    return x + 1"]
    match = match_listing(code, source)
    assert match.firstline == 3
    assert match.lastline == 4
    assert match.score == 1.0


def test_match_ignores_blank_lines_and_indentation_differences():
    """Matching is on stripped lines, skipping blank lines in the source."""
    source = ["def g():", "", "    x = 1", "    return x"]
    code = ["def g():", "x = 1", "return x"]  # re-indented excerpt, blanks dropped
    match = match_listing(code, source)
    assert match.firstline == 1
    assert match.lastline == 4
    assert match.score == 1.0


def test_match_returns_none_when_no_significant_lines_overlap():
    """Code with no lines present in the source yields no match."""
    source = ["import numpy as np", "arr = np.zeros(3)"]
    code = ["totally = 'unrelated'", "other = 2"]
    assert match_listing(code, source) is None


def test_match_partial_overlap_reports_fractional_score():
    """A block where only some lines match reports score < 1.0."""
    source = ["def h():", "    return 42"]
    code = ["def h():", "    return 42", "extra_unmatched_line = True"]
    match = match_listing(code, source)
    assert match.firstline == 1
    assert match.lastline == 2
    assert match.score == 2 / 3


def test_match_anchors_past_an_edited_first_line():
    """When the first listing line was edited, the range starts at the next match."""
    source = ["def f(x):", "    y = x + 1", "    return y"]
    code = [
        "def f(x: int) -> int:",
        "    y = x + 1",
        "    return y",
    ]  # signature edited
    match = match_listing(code, source)
    assert match.firstline == 2  # 'y = x + 1'
    assert match.lastline == 3  # 'return y'
    assert match.score == 2 / 3


# --- coverage ---------------------------------------------------------------


def test_coverage_counts_lines_present_anywhere_ignoring_order():
    """coverage is the fraction of distinct significant lines found anywhere."""
    source = ["return x", "filler", "def f(x):"]  # reordered vs the listing
    code = ["def f(x):", "    return x"]
    assert coverage(code, source) == 1.0


def test_coverage_is_robust_to_one_edited_line():
    """An edited line (e.g. an added type hint) only drops coverage partially."""
    source = ["def f(x):", "    return x + 1", "y = 2"]
    code = ["def f(x: int) -> int:", "    return x + 1", "y = 2"]  # signature edited
    assert coverage(code, source) == 2 / 3


def test_coverage_zero_for_empty_code():
    """Code with no significant lines has zero coverage."""
    assert coverage(["", "   "], ["anything"]) == 0.0


# --- best_source ------------------------------------------------------------


def test_best_source_picks_highest_coverage_file():
    """The candidate with the greatest line coverage wins."""
    code = ["def f(x):", "    return x + 1"]
    sources = {
        "a.py": ["def f(x):", "    return x + 1"],
        "b.py": ["def f(x):", "    return something_else"],
    }
    path, match = best_source(code, sources, min_coverage=0.5)
    assert path == "a.py"
    assert match.firstline == 1 and match.lastline == 2


def test_best_source_returns_none_below_threshold():
    """No candidate clearing the coverage threshold yields (None, None)."""
    code = ["x = 1", "y = 2", "z = 3"]
    sources = {"a.py": ["x = 1", "unrelated", "more"]}  # only 1/3 covered
    path, match = best_source(code, sources, min_coverage=0.5)
    assert path is None
    assert match is None
