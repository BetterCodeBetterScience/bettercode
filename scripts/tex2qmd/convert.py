"""Convert a chapter's LaTeX to Quarto markdown."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow running as a script (uv run python scripts/tex2qmd/convert.py):
# put the scripts/ dir on sys.path so `import tex2qmd.*` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tex2qmd.figures import emit_figure, rewrite_includegraphics  # noqa: E402
from tex2qmd.pandoc_bridge import run_pandoc  # noqa: E402
from tex2qmd.postprocess import framed_to_callout, reinsert_code_blocks  # noqa: E402
from tex2qmd.preprocess import extract_listings, strip_comments, strip_index  # noqa: E402
from tex2qmd.project import (  # noqa: E402
    load_index_body,
    parse_chapter_order,
    qmd_name,
    render_cover,
    render_quarto_yml,
)

TITLE = "Better Code, Better Science"
SUBTITLE = "Software Engineering for Reproducible Science in the Age of AI"
AUTHOR = "Russell A. Poldrack"
REPO_URL = "https://github.com/BetterCodeBetterScience/bettercode"
# Creative Commons license shown as a persistent footer badge on every page.
LICENSE_NAME = "CC BY-NC-ND 4.0"
LICENSE_SLUG = "by-nc-nd/4.0"
# Chapters to omit from the web edition (still present in the print book).
WEB_EXCLUDE = frozenset({"preface"})


def convert_chapter(tex: str, base_dir: Path) -> tuple[str, list[str]]:
    """Convert chapter LaTeX to (qmd text, referenced figure PDF paths)."""
    tex = strip_comments(tex)
    tex = strip_index(tex)
    tex, pdfs = rewrite_includegraphics(tex)
    tex, blocks = extract_listings(tex, base_dir)
    md = run_pandoc(tex)
    qmd = reinsert_code_blocks(md, blocks)
    qmd = framed_to_callout(qmd)
    return qmd, pdfs


def build_book(latex_dir: Path, out_dir: Path) -> None:
    """Convert every chapter in book.tex into a Quarto project under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    book_tex = (latex_dir / "book.tex").read_text()
    targets = parse_chapter_order(book_tex, exclude=WEB_EXCLUDE)

    # A generated cover page is the landing page; every \include (including
    # the preface) becomes its own chapter that follows it.
    (out_dir / "index.qmd").write_text(
        render_cover(TITLE, SUBTITLE, AUTHOR, load_index_body())
    )
    chapter_files: list[str] = ["index.qmd"]
    for target in targets:
        tex = (latex_dir / f"{target}.tex").read_text()
        qmd, pdfs = convert_chapter(tex, latex_dir)
        out_name = qmd_name(target)
        (out_dir / out_name).write_text(qmd)
        chapter_files.append(out_name)
        for pdf in pdfs:
            emit_figure(pdf, latex_dir, out_dir)

    (out_dir / "_quarto.yml").write_text(
        render_quarto_yml(
            chapter_files,
            TITLE,
            AUTHOR,
            subtitle=SUBTITLE,
            repo_url=REPO_URL,
            license_name=LICENSE_NAME,
            license_slug=LICENSE_SLUG,
        )
    )
    shutil.copy(latex_dir / "references.bib", out_dir / "references.bib")
    shutil.copy(
        Path(__file__).with_name("cambridge.csl"), out_dir / "cambridge.csl"
    )


def main() -> None:
    """CLI entry point for building the Quarto book."""
    parser = argparse.ArgumentParser(description="Convert LaTeX book to Quarto")
    parser.add_argument("--latex-dir", default="latex", type=Path)
    parser.add_argument("--out", default="web", type=Path)
    args = parser.parse_args()
    build_book(args.latex_dir, args.out)


if __name__ == "__main__":
    main()
