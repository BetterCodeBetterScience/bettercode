"""Invoke pandoc to convert cleaned LaTeX to markdown."""

from __future__ import annotations

import subprocess


def run_pandoc(latex: str) -> str:
    """Convert a LaTeX string to markdown via pandoc.

    Raises:
        RuntimeError: If pandoc fails; the message carries pandoc's stderr.
    """
    try:
        proc = subprocess.run(
            ["pandoc", "--from", "latex", "--to", "markdown", "--wrap=preserve"],
            input=latex,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"pandoc failed (exit {err.returncode}):\n{err.stderr.strip()}"
        ) from err
    return proc.stdout
