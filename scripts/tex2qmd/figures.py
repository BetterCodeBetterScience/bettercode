"""Convert figure PDFs to SVG and rewrite \\includegraphics references."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_INCLUDE = re.compile(
    r"\\includegraphics(?:\[(?P<opts>[^\]]*)\])?\{(?P<path>[^}]*)\}"
)


def svg_name(pdf_relpath: str) -> str:
    """Map a .pdf figure path to its .svg sibling path."""
    return re.sub(r"\.pdf$", ".svg", pdf_relpath)


def rewrite_includegraphics(tex: str) -> tuple[str, list[str]]:
    """Rewrite includegraphics PDF paths to SVG; return (text, pdf paths)."""
    pdfs: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        path = match.group("path")
        if not path.endswith(".pdf"):
            return match.group(0)
        pdfs.append(path)
        opts = match.group("opts")
        opts_str = f"[{opts}]" if opts is not None else ""
        return f"\\includegraphics{opts_str}{{{svg_name(path)}}}"

    return _INCLUDE.sub(_repl, tex), pdfs


def convert_pdf_to_svg(pdf_path: Path, out_svg: Path) -> None:
    """Convert a single PDF figure to SVG using pdftocairo."""
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftocairo", "-svg", str(pdf_path), str(out_svg)],
        check=True,
    )
