"""Preprocess LaTeX chapter text before handing it to pandoc."""

from __future__ import annotations

import re
from pathlib import Path

from tex2qmd.listings import language_for_style

_INDEX = re.compile(r"\\index\{[^{}]*\}")

#: Marks a `fullbox` title so it survives pandoc as ordinary paragraph text;
#: `postprocess.boxes_to_callouts` turns it back into the callout's heading.
BOX_TITLE_TOKEN = "TEX2QMDBOXTITLE"

_INPUT = re.compile(r"\\input\{(?P<path>[^}]+)\}")


def resolve_inputs(tex: str, base_dir: Path) -> str:
    """Inline every ``\\input{...}`` file, recursively, relative to base_dir.

    Pandoc reads the chapter from stdin and so cannot resolve ``\\input`` itself;
    left alone the command (and the whole fragment behind it) is dropped
    silently. A missing target raises rather than losing content.
    """

    def _repl(match: re.Match[str]) -> str:
        name = match.group("path")
        path = base_dir / (name if name.endswith(".tex") else f"{name}.tex")
        return resolve_inputs(path.read_text(), base_dir)

    return _INPUT.sub(_repl, tex)


# Pandoc parses `\noindent` as taking one argument, so `\noindent{prose}` eats
# the entire paragraph. Indentation control is meaningless in markdown, so the
# command is dropped and any group it was holding is unwrapped.
_NOINDENT_GROUP = re.compile(r"\\noindent[ \t]*\{")
_NOINDENT_BARE = re.compile(r"\\noindent(?![a-zA-Z])[ \t]*")


def _matching_brace(text: str, start: int) -> int:
    """Return the index of the `}` matching the `{` at start, or -1 if unbalanced."""
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def unwrap_noindent(tex: str) -> str:
    """Remove every ``\\noindent``, unwrapping the braced group it wrapped."""
    pos = 0
    while (match := _NOINDENT_GROUP.search(tex, pos)) is not None:
        close = _matching_brace(tex, match.end() - 1)
        if close < 0:  # unbalanced braces: leave this occurrence to the bare pass
            pos = match.end()
            continue
        tex = tex[: match.start()] + tex[match.end() : close] + tex[close + 1 :]
        pos = match.start()
    return _NOINDENT_BARE.sub("", tex)


# `fullbox` is a book-specific environment, so pandoc keeps it as a `::: fullbox`
# div and discards its optional `[title]`. Moving the title into the body as a
# tokenized paragraph lets pandoc render its markup (`\textit{...}`, `\texttt{...}`).
_BOX_OPEN = re.compile(r"\\begin\{fullbox\}(?:\[(?P<title>.*)\])?[ \t]*$", re.MULTILINE)


def mark_box_titles(tex: str, chapter: int) -> str:
    """Move each ``fullbox`` title into the box body as a numbered token.

    Boxes are numbered ``chapter.n`` as the LaTeX ``fullbox`` counter does, so
    chapter prose referring to "Box 5.1" still resolves in the web edition.
    """
    counter = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        label = f"Box {chapter}.{counter}"
        title = match.group("title")
        heading = f"{label} --- {title}" if title else label
        return f"\\begin{{fullbox}}\n\n{BOX_TITLE_TOKEN}{{{heading}}}\n"

    return _BOX_OPEN.sub(_repl, tex)


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


# Some LaTeX listings embed `$...$` math (e.g. `%timeit` output
# `103 $\mu$s $\pm$ 759 ns`), which a verbatim code block would show literally.
# These unambiguous math tokens are converted back to unicode. Full-token keys
# (not a bare `$`) mean a literal shell prompt like `$ git init` is untouched.
_LISTING_MATH = {
    r"$\mu$": "µ",
    r"$\pm$": "±",
    r"$\times$": "×",
    r"$\cdot$": "·",
    r"$\leq$": "≤",
    r"$\geq$": "≥",
    r"$\infty$": "∞",
    r"$\ldots$": "…",
}


def unescape_listing_math(code: str) -> str:
    """Convert common `$...$` LaTeX math (from a mathescape listing) to unicode."""
    for tex, unicode_char in _LISTING_MATH.items():
        code = code.replace(tex, unicode_char)
    return code


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
        code = unescape_listing_math(slice_lines(text, first, last))
        blocks.append(render_code_block(code, language))
        return _token(len(blocks) - 1)

    def _inline_repl(match: re.Match[str]) -> str:
        opts = parse_listing_options(match.group("opts") or "")
        language = language_for_style(opts.get("style", ""))
        code = unescape_listing_math(match.group("body").rstrip("\n"))
        blocks.append(render_code_block(code, language))
        return _token(len(blocks) - 1)

    tex = _INPUT_LISTING.sub(_input_repl, tex)
    tex = _INLINE_LISTING.sub(_inline_repl, tex)
    return tex, blocks
