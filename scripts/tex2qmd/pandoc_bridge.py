"""Invoke pandoc to convert cleaned LaTeX to markdown."""

from __future__ import annotations

import subprocess


def run_pandoc(latex: str) -> str:
    """Convert a LaTeX string to markdown via pandoc."""
    proc = subprocess.run(
        ["pandoc", "--from", "latex", "--to", "markdown", "--wrap=preserve"],
        input=latex,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout
