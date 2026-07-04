import pytest
from tex2qmd.postprocess import (
    framed_to_callout,
    reinsert_code_blocks,
    rewrite_figures,
    rewrite_tables,
)


FIGURE = (
    '<figure id="XPoll-fig" data-latex-placement="!htbp">\n'
    '<img src="figures/01/XPoll.svg" style="width:50.0%" />\n'
    "<figcaption>A social media poll.</figcaption>\n"
    "</figure>"
)


def test_rewrite_figures_emits_quarto_crossref_div():
    """A raw <figure> becomes a Quarto crossref div with a fig- label."""
    out = rewrite_figures(FIGURE)
    assert "::: {#fig-XPoll}" in out
    assert "![](figures/01/XPoll.svg){width=50.0%}" in out
    assert "A social media poll." in out
    assert out.rstrip().endswith(":::")
    # the raw HTML figure is gone
    assert "<figure" not in out
    assert "<figcaption" not in out


def test_rewrite_figures_preserves_inline_html_caption():
    """Inline HTML inside a caption (code, links, citation spans) is kept."""
    fig = (
        '<figure id="ctx-fig" data-latex-placement="!htbp">\n'
        '<img src="figures/05/ctx.svg" style="width:90.0%" />\n'
        "<figcaption>The <code>/context</code> command, see "
        '<span class="citation" data-cites="Wei:2023aa"></span>.</figcaption>\n'
        "</figure>"
    )
    out = rewrite_figures(fig)
    assert "<code>/context</code>" in out
    assert '<span class="citation" data-cites="Wei:2023aa"></span>' in out


def test_rewrite_figures_collapses_multiline_caption():
    """A caption spanning several lines becomes a single caption line."""
    fig = (
        '<figure id="ide-fig" data-latex-placement="!htbp">\n'
        '<img src="figures/03/ide.svg" style="width:75.0%" />\n'
        "<figcaption>First line.\nSecond line.\nThird line.</figcaption>\n"
        "</figure>"
    )
    out = rewrite_figures(fig)
    assert "First line. Second line. Third line." in out


def test_rewrite_figures_handles_missing_width():
    """A figure image without a width style still converts."""
    fig = (
        '<figure id="plain-fig">\n'
        '<img src="figures/x.svg" />\n'
        "<figcaption>Cap.</figcaption>\n"
        "</figure>"
    )
    out = rewrite_figures(fig)
    assert "::: {#fig-plain}" in out
    assert "![](figures/x.svg)" in out


def test_rewrite_figures_rewrites_reference_to_crossref():
    """An in-text figure reference becomes a Quarto @fig- cross-reference."""
    md = (
        'shown in Figure [1.1](#XPoll-fig){reference-type="ref" '
        'reference="XPoll-fig"}.'
    )
    out = rewrite_figures(md)
    assert out == "shown in @fig-XPoll."


def test_rewrite_figures_leaves_table_references_untouched():
    """References to tables (not figures) are not rewritten by rewrite_figures."""
    md = (
        'in Table [1.1](#data-table){reference-type="ref" '
        'reference="data-table"}.'
    )
    assert rewrite_figures(md) == md


def test_rewrite_tables_relabels_caption_to_tbl_prefix():
    """A pandoc table caption label `{#id-table}` becomes a Quarto `{#tbl-id}`."""
    md = "  : Example of wide tabular data {#wide-data-table}\n"
    out = rewrite_tables(md)
    assert "{#tbl-wide-data}" in out
    assert "{#wide-data-table}" not in out


def test_rewrite_tables_relabels_multi_hyphen_id():
    """The `-table` suffix (not every hyphen) is what gets rewritten."""
    md = "  : Untidy data {#untidy-multiple-variables-table}\n"
    out = rewrite_tables(md)
    assert "{#tbl-untidy-multiple-variables}" in out


def test_rewrite_tables_rewrites_reference_to_crossref():
    """An in-text table reference becomes a Quarto @tbl- cross-reference."""
    md = (
        'shown in Table [1.1](#wide-data-table){reference-type="ref" '
        'reference="wide-data-table"}.'
    )
    out = rewrite_tables(md)
    assert out == "shown in @tbl-wide-data."


def test_rewrite_tables_leaves_figure_references_untouched():
    """References to figures (not tables) are not rewritten by rewrite_tables."""
    md = (
        'in Figure [1.1](#XPoll-fig){reference-type="ref" '
        'reference="XPoll-fig"}.'
    )
    assert rewrite_tables(md) == md


def test_reinsert_single_block():
    """A token line is replaced by its fenced code block."""
    md = "Before.\n\nTEX2QMDCODEBLOCK0\n\nAfter."
    blocks = ["```python\nprint(1)\n```"]
    out = reinsert_code_blocks(md, blocks)
    assert "```python\nprint(1)\n```" in out
    assert "TEX2QMDCODEBLOCK0" not in out


def test_reinsert_multiple_blocks_in_order():
    """Multiple tokens map to their indexed blocks."""
    md = "TEX2QMDCODEBLOCK0\n\ntext\n\nTEX2QMDCODEBLOCK1"
    blocks = ["```bash\nls\n```", "```python\nx=1\n```"]
    out = reinsert_code_blocks(md, blocks)
    assert out.index("ls") < out.index("x=1")


def test_reinsert_missing_block_raises():
    """A token with no matching block is an error."""
    md = "TEX2QMDCODEBLOCK5"
    with pytest.raises(ValueError):
        reinsert_code_blocks(md, [])


def test_reinsert_unreferenced_block_raises():
    """A block that no token references is an error."""
    with pytest.raises(ValueError):
        reinsert_code_blocks("TEX2QMDCODEBLOCK0", ["```python\nx=1\n```", "```bash\nls\n```"])


def test_framed_to_callout_converts_opening_fence():
    """A pandoc `::: framed` div opens a Quarto note callout."""
    md = "::: framed\n**Tip**\\\nBe clear.\n:::"
    out = framed_to_callout(md)
    assert "::: {.callout-note}" in out
    assert "::: framed" not in out
    assert out.count(":::") == 2  # opening callout + closing fence preserved


def test_framed_to_callout_leaves_other_divs_untouched():
    """Non-framed fenced divs are not altered."""
    md = "::: {.column-margin}\nAside.\n:::"
    assert framed_to_callout(md) == md
