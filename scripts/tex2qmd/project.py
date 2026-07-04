"""Build the Quarto book project configuration from book.tex."""

from __future__ import annotations

import re
from collections.abc import Container
from pathlib import Path

_INCLUDE = re.compile(r"^\s*\\include\{(?P<target>[^}]+)\}", re.MULTILINE)

#: Editable landing-page prose, shipped alongside the pipeline. Edit this file
#: to change the text shown on the site's index page (no code change needed).
INDEX_BODY_FILE = Path(__file__).with_name("index_body.md")


def parse_chapter_order(
    book_tex: str, exclude: Container[str] = frozenset()
) -> list[str]:
    """Return \\include targets in order, ignoring comments and excluded targets."""
    targets: list[str] = []
    for line in book_tex.splitlines():
        if line.lstrip().startswith("%"):
            continue
        match = _INCLUDE.match(line)
        if match and match.group("target") not in exclude:
            targets.append(match.group("target"))
    return targets


def qmd_name(include_target: str) -> str:
    """Map an \\include target to a .qmd filename."""
    return f"{include_target}.qmd"


def render_navbar(repo_url: str) -> str:
    """Render the book navbar tools (source-code + report-an-issue links)."""
    return f"""  navbar:
    right:
      - icon: github
        href: {repo_url}
        aria-label: Source code on GitHub
      - icon: bug
        text: "Report an issue"
        href: {repo_url}/issues/new
"""


def render_quarto_yml(
    chapter_files: list[str],
    title: str,
    author: str,
    subtitle: str = "",
    repo_url: str = "",
) -> str:
    """Render the _quarto.yml contents for the HTML book."""
    chapter_lines = "\n".join(f"    - {name}" for name in chapter_files)
    subtitle_line = f'  subtitle: "{subtitle}"\n' if subtitle else ""
    navbar_block = render_navbar(repo_url) if repo_url else ""
    return f"""project:
  type: book
  output-dir: ../docs

book:
  title: "{title}"
{subtitle_line}  author: "{author}"
  search: true
{navbar_block}  chapters:
{chapter_lines}

bibliography: references.bib
csl: cambridge.csl
reference-section-title: "References"

format:
  html:
    theme: cosmo
    toc: true
"""


def load_index_body(path: Path | None = None) -> str:
    """Return the editable landing-page prose (defaults to INDEX_BODY_FILE)."""
    return (path or INDEX_BODY_FILE).read_text()


def render_cover(title: str, subtitle: str, author: str, body: str) -> str:
    """Render the landing-page index.qmd for the HTML book.

    The title, subtitle, and author are rendered by Quarto's title block from
    the project config; ``body`` supplies the editable prose beneath it (see
    INDEX_BODY_FILE), so those fields are not repeated in the text.
    """
    return f"""---
title: "{title}"
subtitle: "{subtitle}"
author: "{author}"
---

{body.rstrip()}
"""
