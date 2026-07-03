"""Build the Quarto book project configuration from book.tex."""

from __future__ import annotations

import re

_INCLUDE = re.compile(r"^\s*\\include\{(?P<target>[^}]+)\}", re.MULTILINE)


def parse_chapter_order(book_tex: str) -> list[str]:
    """Return \\include targets in order, ignoring commented-out lines."""
    targets: list[str] = []
    for line in book_tex.splitlines():
        if line.lstrip().startswith("%"):
            continue
        match = _INCLUDE.match(line)
        if match:
            targets.append(match.group("target"))
    return targets


def qmd_name(include_target: str) -> str:
    """Map an \\include target to a .qmd filename."""
    return f"{include_target}.qmd"


def render_quarto_yml(
    chapter_files: list[str], title: str, author: str, subtitle: str = ""
) -> str:
    """Render the _quarto.yml contents for the HTML book."""
    chapter_lines = "\n".join(f"    - {name}" for name in chapter_files)
    subtitle_line = f'  subtitle: "{subtitle}"\n' if subtitle else ""
    return f"""project:
  type: book
  output-dir: ../docs

book:
  title: "{title}"
{subtitle_line}  author: "{author}"
  search: true
  chapters:
{chapter_lines}

bibliography: references.bib
csl: cambridge.csl
reference-section-title: "References"

format:
  html:
    theme: cosmo
    toc: true
"""


def render_cover(title: str, subtitle: str, author: str) -> str:
    """Render the landing-page index.qmd for the HTML book.

    The title, subtitle, and author are rendered by Quarto's title block from
    the project config, so the body only carries a short welcome to avoid
    repeating them.
    """
    return f"""---
title: "{title}"
subtitle: "{subtitle}"
author: "{author}"
---

Welcome to the web edition of *{title}*.

This site is generated automatically from the book's LaTeX source. Use the
sidebar to navigate the chapters, or the search box to find specific topics.
"""
