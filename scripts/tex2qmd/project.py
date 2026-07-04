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


def render_page_footer(license_name: str, license_slug: str) -> str:
    """Render a persistent page footer with a Creative Commons license badge.

    ``license_slug`` is the CC path segment (e.g. ``by-nc-nd/4.0``) used to
    build both the badge image and the human-readable license page.
    """
    license_url = f"https://creativecommons.org/licenses/{license_slug}/"
    badge_url = f"https://licensebuttons.net/l/{license_slug}/88x31.png"
    return f"""  page-footer:
    center: |
      [![{license_name}]({badge_url})]({license_url})
"""


def render_quarto_yml(
    chapter_files: list[str],
    title: str,
    author: str,
    subtitle: str = "",
    repo_url: str = "",
    license_name: str = "",
    license_slug: str = "",
) -> str:
    """Render the _quarto.yml contents for the HTML book."""
    chapter_lines = "\n".join(f"    - {name}" for name in chapter_files)
    subtitle_line = f'  subtitle: "{subtitle}"\n' if subtitle else ""
    navbar_block = render_navbar(repo_url) if repo_url else ""
    footer_block = (
        render_page_footer(license_name, license_slug) if license_slug else ""
    )
    return f"""project:
  type: book
  output-dir: ../docs

book:
  title: "{title}"
{subtitle_line}  author: "{author}"
  search: true
{navbar_block}{footer_block}  chapters:
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


def render_cover(subtitle: str, author: str, body: str) -> str:
    """Render the landing-page index.qmd for the HTML book.

    The book title is rendered by Quarto's title block from the project config,
    so no front-matter ``title`` is emitted here: a front-matter title would make
    Quarto count the landing page as numbered chapter 1 (pushing the first real
    chapter to 2). An explicit unnumbered heading keeps the cover out of the
    chapter count while still displaying subtitle, author, and the editable body.
    """
    return f"""---
subtitle: "{subtitle}"
author: "{author}"
---

# Welcome {{.unnumbered}}

{body.rstrip()}
"""
