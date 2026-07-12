"""Fill and verify the grayscale figure tree for the print build.

Notebooks emit purpose-designed grayscale figures into latex/gray/figures/.
This tool covers the remaining figures referenced by the .tex chapters
(hand-made diagrams, screenshots) by color-converting them with ghostscript
(PDF) or Pillow (PNG) into the gray tree, and verifies that every referenced
figure has a gray variant. Auto-converted files are reported so the author
can review their legibility.

Usage:
    uv run python scripts/build_gray_figures.py [--latex-dir latex] [--check]
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics\*?(?:\[[^\]]*\])?\{([^}]+)\}")


def _uncommented(line: str) -> str:
    """The portion of a LaTeX source line before any % comment."""
    match = re.search(r"(?<!\\)%", line)
    return line[: match.start()] if match else line


def find_figure_refs(latex_dir: Path) -> set[Path]:
    """All \\includegraphics targets in the .tex files, as latex_dir-relative paths."""
    refs = set()
    for tex in latex_dir.glob("*.tex"):
        for line in tex.read_text().splitlines():
            for match in INCLUDEGRAPHICS_RE.finditer(_uncommented(line)):
                refs.add(Path(match.group(1)))
    return refs


def convert_pdf_to_gray(src: Path, dest: Path) -> None:
    """Convert a PDF to grayscale with ghostscript, preserving vector content."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gs",
            "-sDEVICE=pdfwrite",
            "-sColorConversionStrategy=Gray",
            "-dProcessColorModel=/DeviceGray",
            "-dCompatibilityLevel=1.4",
            "-dNOPAUSE",
            "-dBATCH",
            "-dQUIET",
            f"-sOutputFile={dest}",
            str(src),
        ],
        check=True,
    )


def convert_png_to_gray(src: Path, dest: Path) -> None:
    """Convert a raster image to grayscale with Pillow."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("L").save(dest)


@dataclass
class GrayTreeReport:
    """Outcome of a gap-fill pass over the gray figure tree."""

    converted: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    missing_sources: list[Path] = field(default_factory=list)


def fill_gray_tree(latex_dir: Path) -> GrayTreeReport:
    """Auto-convert referenced figures that lack a gray variant into gray/."""
    converters = {".pdf": convert_pdf_to_gray, ".png": convert_png_to_gray}
    report = GrayTreeReport()
    for ref in sorted(find_figure_refs(latex_dir)):
        src = latex_dir / ref
        dest = latex_dir / "gray" / ref
        if dest.exists():
            report.skipped.append(ref)
        elif not src.exists():
            report.missing_sources.append(ref)
        else:
            converters[src.suffix.lower()](src, dest)
            report.converted.append(ref)
    return report


def verify_complete(latex_dir: Path) -> list[Path]:
    """Referenced figures that have no variant in the gray tree."""
    return sorted(
        ref
        for ref in find_figure_refs(latex_dir)
        if not (latex_dir / "gray" / ref).exists()
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latex-dir", type=Path, default=Path("latex"))
    parser.add_argument(
        "--check", action="store_true", help="only verify completeness, convert nothing"
    )
    args = parser.parse_args()

    if not args.check:
        report = fill_gray_tree(args.latex_dir)
        for label, paths in [
            ("auto-converted (review for legibility)", report.converted),
            ("kept existing gray variant", report.skipped),
            ("MISSING color source", report.missing_sources),
        ]:
            print(f"{label}: {len(paths)}")
            for path in paths:
                print(f"  {path}")

    missing = verify_complete(args.latex_dir)
    if missing:
        print(f"INCOMPLETE gray tree, {len(missing)} referenced figures missing:")
        for path in missing:
            print(f"  {path}")
        return 1
    print("gray tree complete: every referenced figure has a gray variant")
    return 0


if __name__ == "__main__":
    sys.exit(main())
