"""Reinsert extracted code blocks into pandoc's markdown output."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"TEX2QMDCODEBLOCK(?P<index>\d+)")


def reinsert_code_blocks(md: str, blocks: list[str]) -> str:
    """Replace each code placeholder token with its fenced code block."""
    seen: set[int] = set()

    def _repl(match: re.Match[str]) -> str:
        index = int(match.group("index"))
        if index >= len(blocks):
            raise ValueError(f"No code block for token index {index}")
        seen.add(index)
        return blocks[index]

    out = _TOKEN.sub(_repl, md)
    if len(seen) != len(blocks):
        missing = set(range(len(blocks))) - seen
        raise ValueError(f"Code blocks never referenced: {sorted(missing)}")
    return out


def framed_to_callout(md: str) -> str:
    """Convert pandoc `::: framed` fenced divs to Quarto note callouts."""
    return re.sub(r"^::: +framed\s*$", "::: {.callout-note}", md, flags=re.MULTILINE)


# Pandoc renders LaTeX figure environments as raw HTML <figure> blocks whose
# numbering it bakes in per-chapter (every chapter restarts at 1.x). Converting
# them to Quarto crossref figures hands numbering to Quarto, which numbers them
# by the chapter they render in (e.g. 10.1 in chapter 10).
_FIGURE_BLOCK = re.compile(
    r'<figure id="(?P<id>[^"]+)"[^>]*>\s*'
    r'<img src="(?P<src>[^"]+)"(?:\s+style="width:(?P<width>[\d.]+)%")?\s*/?>\s*'
    r"<figcaption>(?P<cap>.*?)</figcaption>\s*"
    r"</figure>",
    re.DOTALL,
)

# A pandoc figure cross-reference, e.g. `Figure [1.1](#foo-fig){reference-type=
# "ref" reference="foo-fig"}`. A leading "Figure"/"Fig." word (with its trailing
# separator) is consumed so Quarto can regenerate it; matching only `-fig`
# targets leaves table references untouched.
_FIGURE_REF = re.compile(
    r"(?:(?:Figures?|Fig\.?)[ \u00a0~]*)?"
    r"\[[0-9.]+\]\(#(?P<id>[A-Za-z0-9_-]+-fig)\)"
    r'\{reference-type="ref"\s+reference="(?P=id)"\}'
)


def _fig_label(figure_id: str) -> str:
    """Map a pandoc figure id (e.g. `XPoll-fig`) to a Quarto label (`fig-XPoll`)."""
    return "fig-" + re.sub(r"-fig$", "", figure_id)


def _figure_block_to_div(match: re.Match[str]) -> str:
    label = _fig_label(match.group("id"))
    width = match.group("width")
    width_attr = f"{{width={width}%}}" if width else ""
    caption = " ".join(match.group("cap").split())
    return (
        f"::: {{#{label}}}\n"
        f"![]({match.group('src')}){width_attr}\n\n"
        f"{caption}\n"
        f":::"
    )


def rewrite_figures(md: str) -> str:
    """Convert raw <figure> HTML blocks to Quarto crossref figures and fix refs.

    Figures become `::: {#fig-…}` divs and in-text `Figure [n.n](#…-fig)`
    references become `@fig-…`, so Quarto owns figure numbering per chapter.
    """
    md = _FIGURE_BLOCK.sub(_figure_block_to_div, md)
    md = _FIGURE_REF.sub(lambda m: "@" + _fig_label(m.group("id")), md)
    return md
