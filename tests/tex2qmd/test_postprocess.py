import pytest
from tex2qmd.postprocess import framed_to_callout, reinsert_code_blocks


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
