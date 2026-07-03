"""Preprocess LaTeX chapter text before handing it to pandoc."""

from __future__ import annotations

from tex2qmd.listings import language_for_style


def slice_lines(text: str, firstline: int | None, lastline: int | None) -> str:
    """Return lines [firstline, lastline] (1-indexed, inclusive) of text."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    start = (firstline - 1) if firstline else 0
    end = lastline if lastline else len(lines)
    return "\n".join(lines[start:end])


def parse_listing_options(opts: str) -> dict[str, str]:
    """Parse a lstlisting/lstinputlisting `[...]` option string into a dict."""
    result: dict[str, str] = {}
    for part in opts.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        result[key.strip()] = value.strip()
    return result


def render_code_block(code: str, language: str) -> str:
    """Wrap code in a fenced block tagged with language."""
    return f"```{language}\n{code.rstrip(chr(10))}\n```"
