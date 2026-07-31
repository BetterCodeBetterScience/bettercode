import pytest
from tex2qmd.postprocess import (
    boxes_to_callouts,
    framed_to_callout,
    modelquotes_to_blockquotes,
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


def test_boxes_to_callouts_converts_fullbox_div():
    """A pandoc `::: fullbox` div becomes a Quarto note callout."""
    md = "::: fullbox\nBody.\n:::"
    out = boxes_to_callouts(md)
    assert "::: {.callout-note}" in out
    assert "fullbox" not in out
    assert "Body." in out


def test_boxes_to_callouts_renders_title_token_as_heading():
    """The title token planted by mark_box_titles becomes the callout's heading."""
    md = ":::: fullbox\nTEX2QMDBOXTITLE*GPT-4*'s hurdle\n\nBody.\n::::"
    out = boxes_to_callouts(md)
    assert "## *GPT-4*'s hurdle" in out
    assert "TEX2QMDBOXTITLE" not in out


def test_boxes_to_callouts_preserves_fence_length():
    """A nested box keeps its longer opening fence so it still matches its closer."""
    md = ":::: fullbox\nBody.\n\n::: modelquote\nQuoted.\n:::\n::::"
    out = boxes_to_callouts(md)
    assert ":::: {.callout-note}" in out
    assert "::: modelquote" in out  # inner div is left for its own rewriter


def test_boxes_to_callouts_leaves_other_divs_untouched():
    """Non-fullbox fenced divs are not altered."""
    md = "::: framed\nTip.\n:::"
    assert boxes_to_callouts(md) == md


def test_modelquotes_to_blockquotes_prefixes_body():
    """A `::: modelquote` div becomes a markdown blockquote."""
    md = "::: modelquote\nFirst line.\nSecond line.\n:::"
    assert modelquotes_to_blockquotes(md) == "> First line.\n> Second line."


def test_modelquotes_to_blockquotes_keeps_blank_lines_quoted():
    """A blank line inside the quote stays inside the blockquote."""
    md = "::: modelquote\nPara one.\n\nPara two.\n:::"
    assert modelquotes_to_blockquotes(md) == "> Para one.\n>\n> Para two."


def test_modelquotes_to_blockquotes_converts_each_of_several():
    """Several model quotes in one chapter are each converted."""
    md = "::: modelquote\nA.\n:::\n\nBetween.\n\n::: modelquote\nB.\n:::"
    out = modelquotes_to_blockquotes(md)
    assert out == "> A.\n\nBetween.\n\n> B."


def test_modelquotes_to_blockquotes_survives_inside_longer_fence():
    """A model quote nested in a box is converted without eating the outer fence."""
    md = ":::: fullbox\nIntro.\n\n::: modelquote\nQuoted.\n:::\n::::"
    out = modelquotes_to_blockquotes(md)
    assert out == ":::: fullbox\nIntro.\n\n> Quoted.\n::::"


def test_modelquotes_to_blockquotes_leaves_other_divs_untouched():
    """Non-modelquote fenced divs are not altered."""
    md = "::: framed\nTip.\n:::"
    assert modelquotes_to_blockquotes(md) == md
