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
