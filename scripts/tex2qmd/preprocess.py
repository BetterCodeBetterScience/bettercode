"""Preprocess LaTeX chapter text before handing it to pandoc."""

from __future__ import annotations

import re
from pathlib import Path

from tex2qmd.listings import language_for_style

_INDEX = re.compile(r"\\index\{[^{}]*\}")


def strip_comments(tex: str) -> str:
    """Remove full-line LaTeX comments that lie outside lstlisting environments.

    Lines whose first non-whitespace character is ``%`` are dropped, so
    commented-out commands (e.g. ``% \\lstinputlisting{...}``) are never
    processed. Lines inside a ``lstlisting`` environment are left untouched so
    literal ``%`` in code (such as ``%matplotlib``) survives.
    """
    kept: list[str] = []
    in_listing = False
    for line in tex.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("\\begin{lstlisting}"):
            in_listing = True
        elif stripped.startswith("\\end{lstlisting}"):
            in_listing = False
        elif not in_listing and stripped.startswith("%"):
            continue
        kept.append(line)
    return "\n".join(kept)
_INPUT_LISTING = re.compile(
    r"\\lstinputlisting(?:\[(?P<opts>[^\]]*)\])?\{(?P<path>[^}]*)\}"
)
_INLINE_LISTING = re.compile(
    r"\\begin\{lstlisting\}(?:\[(?P<opts>[^\]]*)\])?\n?(?P<body>.*?)\\end\{lstlisting\}",
    re.DOTALL,
)


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


def strip_index(tex: str) -> str:
    """Remove all \\index{...} commands from the text."""
    return _INDEX.sub("", tex)


def render_code_block(code: str, language: str) -> str:
    """Wrap code in a fenced block tagged with language."""
    return f"```{language}\n{code.rstrip(chr(10))}\n```"


def _token(index: int) -> str:
    """Return the placeholder token for the given listing index."""
    return f"\n\nTEX2QMDCODEBLOCK{index}\n\n"


def extract_listings(tex: str, base_dir: Path) -> tuple[str, list[str]]:
    """Replace all listings with tokens; return (text, fenced code blocks)."""
    blocks: list[str] = []

    def _input_repl(match: re.Match[str]) -> str:
        opts = parse_listing_options(match.group("opts") or "")
        language = language_for_style(opts.get("style", ""))
        path = (base_dir / match.group("path")).resolve()
        text = path.read_text()
        first = int(opts["firstline"]) if "firstline" in opts else None
        last = int(opts["lastline"]) if "lastline" in opts else None
        code = slice_lines(text, first, last)
        blocks.append(render_code_block(code, language))
        return _token(len(blocks) - 1)

    def _inline_repl(match: re.Match[str]) -> str:
        opts = parse_listing_options(match.group("opts") or "")
        language = language_for_style(opts.get("style", ""))
        code = match.group("body").rstrip("\n")
        blocks.append(render_code_block(code, language))
        return _token(len(blocks) - 1)

    tex = _INPUT_LISTING.sub(_input_repl, tex)
    tex = _INLINE_LISTING.sub(_inline_repl, tex)
    return tex, blocks
