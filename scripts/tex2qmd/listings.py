"""Map LaTeX lstlisting style names to Quarto fenced-code languages."""

from __future__ import annotations

LISTING_LANGUAGES: dict[str, str] = {
    "Python": "python",
    "Pythonshort": "python",
    "repl": "python",
    "replshort": "python",
    "shell": "bash",
    "shellshort": "bash",
    "transcript": "text",
    "transcriptshort": "text",
    "R": "r",
    "yaml": "yaml",
    "toml": "toml",
    "json": "json",
    "jsonshort": "json",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "Rust": "rust",
}


def language_for_style(style: str) -> str:
    """Return the fenced-code language for a lstlisting style name."""
    return LISTING_LANGUAGES.get(style, "text")
