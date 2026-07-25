"""Map LaTeX ``lstlisting`` code blocks to their source files.

The book's ``.tex`` chapters embed code in inline ``lstlisting`` blocks that
were originally copied from notebooks (``notebooks/*.py``), library modules
(``src/bettercode/**/*.py``), or tests. This module extracts those blocks and
finds the best-matching source file by normalized line overlap, producing an
"atlas" that supports migrating the inline blocks to ``\\lstinputlisting``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LATEX_DIR = REPO_ROOT / "latex"

# Styles whose contents are program transcripts, not source-file excerpts.
TRANSCRIPT_STYLES = {"shell", "repl"}

# Glob patterns (relative to repo root) for candidate source files.
SOURCE_GLOBS = (
    "notebooks/*.py",
    "src/bettercode/**/*.py",
    "tests/**/*.py",
)

# Minimum line coverage for a listing to be considered sourced from a file.
MIN_COVERAGE = 0.5

_BEGIN = re.compile(r"\\begin\{lstlisting\}(?:\[(?P<opts>[^\]]*)\])?")
_END = re.compile(r"\\end\{lstlisting\}")
_STYLE = re.compile(r"style\s*=\s*(?P<style>[A-Za-z0-9_]+)")
_LANGUAGE = re.compile(r"language\s*=\s*(?P<language>[A-Za-z0-9_]+)")

# Styles/languages that are Python source excerpts worth matching to a file.
PYTHON_STYLES = {"Python", "Pythonshort"}


@dataclass
class Listing:
    """A single ``lstlisting`` block extracted from a ``.tex`` file."""

    style: str | None
    code: list[str]
    begin_line: int
    end_line: int


@dataclass
class Match:
    """A best-match alignment of a listing against one source file."""

    firstline: int
    lastline: int
    score: float
    matched: int
    total: int


def extract_listings(tex_text: str) -> list[Listing]:
    """Extract every ``lstlisting`` block from LaTeX source, in document order."""
    lines = tex_text.split("\n")
    listings: list[Listing] = []
    i = 0
    while i < len(lines):
        begin = _BEGIN.search(lines[i])
        if begin is None:
            i += 1
            continue
        style = _extract_style(begin.group("opts"))
        begin_line = i + 1
        code: list[str] = []
        j = i + 1
        while j < len(lines) and not _END.search(lines[j]):
            code.append(lines[j])
            j += 1
        listings.append(
            Listing(style=style, code=code, begin_line=begin_line, end_line=j + 1)
        )
        i = j + 1
    return listings


def _extract_style(opts: str | None) -> str | None:
    """Pull the ``style=`` (or fallback ``language=``) value out of an option string."""
    if not opts:
        return None
    found = _STYLE.search(opts)
    if found:
        return found.group("style")
    lang = _LANGUAGE.search(opts)
    return lang.group("language") if lang else None


def coverage(code_lines: list[str], source_lines: list[str]) -> float:
    """Fraction of a listing's distinct significant lines present anywhere in source.

    Order-independent and tolerant of edited lines, so it is a robust signal
    for identifying which source file a listing came from.
    """
    code_sig = {s for s in (line.strip() for line in code_lines) if s}
    if not code_sig:
        return 0.0
    src_sig = {s.strip() for s in source_lines if s.strip()}
    return len(code_sig & src_sig) / len(code_sig)


def best_source(
    code_lines: list[str],
    sources: dict[str, list[str]],
    min_coverage: float = 0.5,
) -> tuple[str | None, Match | None]:
    """Pick the source file that best accounts for a listing.

    Candidates are ranked by line coverage (then by contiguous-match score and
    length). Returns ``(None, None)`` when no candidate clears ``min_coverage``.
    """
    best_path: str | None = None
    best_match: Match | None = None
    best_key = (min_coverage, -1.0, -1)
    for path, lines in sources.items():
        cov = coverage(code_lines, lines)
        if cov < min_coverage:
            continue
        match = match_listing(code_lines, lines)
        score = match.score if match else 0.0
        matched = match.matched if match else 0
        key = (cov, score, matched)
        if key > best_key:
            best_key = key
            best_path = path
            best_match = match
    return best_path, best_match


def match_listing(code_lines: list[str], source_lines: list[str]) -> Match | None:
    """Find where a listing's significant lines best align within a source file.

    Compares stripped, non-blank lines in order, skipping blank lines in the
    source. Returns the 1-based source line range spanning the matched region
    and a score (fraction of the listing's significant lines matched), or
    ``None`` when nothing aligns.
    """
    code_sig = [s for s in (line.strip() for line in code_lines) if s]
    if not code_sig:
        return None

    # Index of stripped source lines paired with their 1-based line numbers,
    # excluding blank lines (which we allow the listing to omit).
    src = [(n + 1, s.strip()) for n, s in enumerate(source_lines) if s.strip()]

    # Find the longest in-order contiguous run of listing lines anywhere in the
    # source, anchoring on any listing line so an edited first/last line does
    # not defeat the match.
    best: Match | None = None
    for ci0 in range(len(code_sig)):
        for si0 in range(len(src)):
            if src[si0][1] != code_sig[ci0]:
                continue
            matched = 1
            last_lineno = src[si0][0]
            si = si0 + 1
            for ci in range(ci0 + 1, len(code_sig)):
                if si < len(src) and src[si][1] == code_sig[ci]:
                    last_lineno = src[si][0]
                    matched += 1
                    si += 1
                else:
                    break
            if best is None or matched > best.matched:
                best = Match(
                    firstline=src[si0][0],
                    lastline=last_lineno,
                    score=matched / len(code_sig),
                    matched=matched,
                    total=len(code_sig),
                )
    return best


# --- driver / reporting -----------------------------------------------------


def load_sources(root: Path) -> dict[str, list[str]]:
    """Read every candidate source file, keyed by repo-relative POSIX path."""
    sources: dict[str, list[str]] = {}
    for pattern in SOURCE_GLOBS:
        for path in sorted(root.glob(pattern)):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(root).as_posix()
            sources[rel] = path.read_text(encoding="utf-8").split("\n")
    return sources


def _confidence(cov: float) -> str:
    """Bucket a coverage fraction into a human-readable confidence label."""
    if cov >= 0.9:
        return "high"
    if cov >= 0.7:
        return "medium"
    return "low"


def classify_listing(listing: Listing, sources: dict[str, list[str]]) -> dict:
    """Build the atlas record for one listing (matched, transcript, or unmatched)."""
    code_lines = [line for line in listing.code if line.strip()]
    preview = code_lines[0].strip() if code_lines else ""
    record = {
        "style": listing.style,
        "begin_line": listing.begin_line,
        "end_line": listing.end_line,
        "n_code_lines": len(code_lines),
        "preview": preview,
        "source": None,
        "firstline": None,
        "lastline": None,
        "coverage": None,
        "contiguous_score": None,
        "status": None,
    }
    if listing.style in TRANSCRIPT_STYLES:
        record["status"] = "transcript"
        return record

    # Only Python excerpts (or unstyled blocks, which may be Python) are matched
    # against source files; other markup/languages have no .py source by design.
    if listing.style is not None and listing.style not in PYTHON_STYLES:
        record["status"] = f"non-python ({listing.style})"
        return record

    path, match = best_source(listing.code, sources, MIN_COVERAGE)
    if path is None:
        record["status"] = "unmatched"
        return record

    cov = coverage(listing.code, sources[path])
    record.update(
        source=path,
        firstline=match.firstline if match else None,
        lastline=match.lastline if match else None,
        coverage=round(cov, 3),
        contiguous_score=round(match.score, 3) if match else None,
        status=f"matched ({_confidence(cov)})",
    )
    return record


def build_atlas(latex_dir: Path, sources: dict[str, list[str]]) -> dict:
    """Extract and classify every listing across all chapter ``.tex`` files."""
    chapters = []
    for tex in sorted(latex_dir.glob("book-*.tex")):
        if tex.name == "book.tex":
            continue
        listings = extract_listings(tex.read_text(encoding="utf-8"))
        records = [classify_listing(item, sources) for item in listings]
        for i, rec in enumerate(records, start=1):
            rec["index"] = i
        chapters.append(
            {"file": tex.name, "n_listings": len(records), "listings": records}
        )
    return {
        "candidate_source_count": len(sources),
        "min_coverage": MIN_COVERAGE,
        "chapters": chapters,
    }


def render_markdown(atlas: dict) -> str:
    """Render the atlas as a human-readable markdown report grouped by chapter."""
    lines = [
        "# Listing atlas",
        "",
        "Maps each `lstlisting` block in the chapter `.tex` files to its best-match "
        "source file. Generated by `scripts/listing_atlas.py` — do not edit by hand.",
        "",
        f"- Candidate source files searched: **{atlas['candidate_source_count']}**",
        f"- Minimum line coverage to count as matched: **{atlas['min_coverage']}**",
        "- `transcript` = shell/repl output (no source file by design).",
        "- `non-python (...)` = a non-Python markup/language block (yaml, json, "
        "bash, R, …); no `.py` source expected.",
        "- `unmatched` = a Python/unstyled block where no candidate cleared the "
        "coverage threshold (hand-typed, REPL-pasted, or edited beyond recognition).",
        "- Confidence reflects coverage: high ≥ 0.9, medium ≥ 0.7, low ≥ 0.5.",
        "- `Cov` is the fraction of the listing's lines found anywhere in the "
        "source (file identification). The source line range comes from the longest "
        "contiguous run; when it is much shorter than `Cov` implies, the file is "
        "right but the range is approximate (the listing was reformatted/excerpted).",
        "",
    ]
    totals: dict[str, int] = {}
    for chapter in atlas["chapters"]:
        for rec in chapter["listings"]:
            key = rec["status"].split(" ")[0]
            totals[key] = totals.get(key, 0) + 1
    summary = " · ".join(f"{n} {k}" for k, n in sorted(totals.items()))
    lines.append(f"**Totals:** {summary}")
    lines.append("")

    for chapter in atlas["chapters"]:
        lines.append(f"## {chapter['file']} ({chapter['n_listings']} listings)")
        lines.append("")
        lines.append(
            "| # | .tex lines | Style | Code lines | Source file | Src lines | Cov | Status | First line |"
        )
        lines.append(
            "|---|-----------|-------|-----------|-------------|-----------|-----|--------|-----------|"
        )
        for rec in chapter["listings"]:
            src = rec["source"] or "—"
            srcrange = (
                f"{rec['firstline']}–{rec['lastline']}" if rec["firstline"] else "—"
            )
            cov = "" if rec["coverage"] is None else f"{rec['coverage']:.2f}"
            preview = rec["preview"].replace("|", "\\|")[:60]
            lines.append(
                f"| {rec['index']} | {rec['begin_line']}–{rec['end_line']} | "
                f"{rec['style'] or '—'} | {rec['n_code_lines']} | `{src}` | {srcrange} | "
                f"{cov} | {rec['status']} | `{preview}` |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Generate the listing atlas as JSON and rendered markdown under ``latex/``."""
    sources = load_sources(REPO_ROOT)
    atlas = build_atlas(LATEX_DIR, sources)
    json_path = LATEX_DIR / "listing_atlas.json"
    md_path = LATEX_DIR / "LISTING_ATLAS.md"
    json_path.write_text(json.dumps(atlas, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(atlas), encoding="utf-8")
    print(
        f"Wrote {json_path.relative_to(REPO_ROOT)} and {md_path.relative_to(REPO_ROOT)}"
    )


if __name__ == "__main__":
    main()
