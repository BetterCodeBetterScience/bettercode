"""Convert a chapter's LaTeX to Quarto markdown."""

from __future__ import annotations

from pathlib import Path

from tex2qmd.figures import rewrite_includegraphics
from tex2qmd.pandoc_bridge import run_pandoc
from tex2qmd.postprocess import reinsert_code_blocks
from tex2qmd.preprocess import extract_listings, strip_index


def convert_chapter(tex: str, base_dir: Path) -> tuple[str, list[str]]:
    """Convert chapter LaTeX to (qmd text, referenced figure PDF paths)."""
    tex = strip_index(tex)
    tex, pdfs = rewrite_includegraphics(tex)
    tex, blocks = extract_listings(tex, base_dir)
    md = run_pandoc(tex)
    qmd = reinsert_code_blocks(md, blocks)
    return qmd, pdfs
