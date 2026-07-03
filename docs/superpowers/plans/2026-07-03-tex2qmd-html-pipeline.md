# tex2qmd LaTeX → Quarto HTML Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a rerunnable Python pipeline that mechanically converts the book's LaTeX source into a Quarto project rendered as an HTML book for GitHub Pages, keeping LaTeX as the single source of truth.

**Architecture:** A `scripts/tex2qmd/` package converts one chapter at a time. For each chapter: (1) preprocess extracts code listings into placeholder tokens, strips print-only markup, and rewrites figure references; (2) pandoc converts the cleaned LaTeX (prose, math, citations, cross-refs) to markdown; (3) postprocess reinserts fenced code blocks and normalizes to Quarto idioms. A project module builds `_quarto.yml` from `book.tex`'s `\include` order. Output lands in a gitignored top-level `web/` directory, rendered by `quarto render`.

**Tech Stack:** Python 3.13 (stdlib only — `re`, `pathlib`, `subprocess`, `argparse`), pandoc 3.10, quarto 1.6, `pdftocairo` (poppler) for PDF→SVG, pytest, `uv run`.

## Global Constraints

- Package management: use `uv`; run all local commands via `uv run`.
- `__init__.py` files MUST be empty — no code, no imports.
- TDD is mandatory: write and commit the failing test before implementation (RED → GREEN → Refactor). Never weaken a test to pass.
- Tests are functions, not classes; use pytest fixtures for shared resources; test functions get docstrings but NO type hints.
- Non-test functions: Google-style docstrings, type hints in the signature only, prefer built-in generics (`list[str]`).
- NEVER edit narrative prose in the `.tex` files. The pipeline only reads them.
- Generated `web/` content is gitignored and never hand-edited.
- Paths in `\lstinputlisting`/`\includegraphics` are resolved relative to the `latex/` directory.
- Run tests from the repo root: `/Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode`.

---

### Task 1: Scaffold package + listing style→language map

**Files:**
- Create: `scripts/tex2qmd/__init__.py` (empty)
- Create: `scripts/tex2qmd/listings.py`
- Create: `tests/tex2qmd/__init__.py` (empty)
- Test: `tests/tex2qmd/test_listings.py`

**Interfaces:**
- Produces: `LISTING_LANGUAGES: dict[str, str]`; `language_for_style(style: str) -> str` (returns mapped fence language, `"text"` for unknown/empty).

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_listings.py
import pytest
from tex2qmd.listings import language_for_style


@pytest.mark.parametrize(
    "style,expected",
    [
        ("Python", "python"),
        ("Pythonshort", "python"),
        ("repl", "python"),
        ("replshort", "python"),
        ("shell", "bash"),
        ("shellshort", "bash"),
        ("transcript", "text"),
        ("transcriptshort", "text"),
        ("R", "r"),
        ("yaml", "yaml"),
        ("toml", "toml"),
        ("json", "json"),
        ("jsonshort", "json"),
        ("dockerfile", "dockerfile"),
        ("makefile", "makefile"),
        ("Rust", "rust"),
        ("", "text"),
        ("NonExistentStyle", "text"),
    ],
)
def test_language_for_style(style, expected):
    """Each listing style maps to the expected fenced-code language."""
    assert language_for_style(style) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_listings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tex2qmd'` (or import error).

Note: add `scripts` to the import path. Create `tests/tex2qmd/conftest.py`:

```python
# tests/tex2qmd/conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
```

Re-run; expected FAIL — `cannot import name 'language_for_style'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/tex2qmd/listings.py
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
```

Create empty files: `scripts/tex2qmd/__init__.py`, `tests/tex2qmd/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_listings.py -v`
Expected: PASS (18 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/__init__.py scripts/tex2qmd/listings.py tests/tex2qmd/
git commit -m "feat(tex2qmd): add listing style to language map"
```

---

### Task 2: Line-range slicing + `\lstinputlisting` resolution

**Files:**
- Create: `scripts/tex2qmd/preprocess.py`
- Test: `tests/tex2qmd/test_preprocess.py`

**Interfaces:**
- Consumes: `language_for_style` from `tex2qmd.listings`.
- Produces:
  - `slice_lines(text: str, firstline: int | None, lastline: int | None) -> str` (1-indexed, inclusive, matching LaTeX `firstline`/`lastline` semantics).
  - `parse_listing_options(opts: str) -> dict[str, str]` (parses the `[...]` option string into a dict; keys `style`, `firstline`, `lastline`, etc.).
  - `render_code_block(code: str, language: str) -> str` (returns a fenced block: `` ```{language}\n{code}\n``` ``, trailing newline stripped from code).

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_preprocess.py
from tex2qmd.preprocess import slice_lines, parse_listing_options, render_code_block


def test_slice_lines_inclusive_range():
    """firstline/lastline slice is 1-indexed and inclusive."""
    text = "a\nb\nc\nd\ne\n"
    assert slice_lines(text, 2, 4) == "b\nc\nd"


def test_slice_lines_no_bounds_returns_all():
    """No firstline/lastline returns the full text (trailing newline trimmed)."""
    text = "a\nb\nc\n"
    assert slice_lines(text, None, None) == "a\nb\nc"


def test_slice_lines_firstline_only():
    """firstline with no lastline slices to end."""
    text = "a\nb\nc\nd\n"
    assert slice_lines(text, 3, None) == "c\nd"


def test_parse_listing_options():
    """Option string parses into a key/value dict."""
    opts = "style=Python, firstline=8, lastline=16"
    assert parse_listing_options(opts) == {
        "style": "Python",
        "firstline": "8",
        "lastline": "16",
    }


def test_parse_listing_options_empty():
    """Empty option string yields an empty dict."""
    assert parse_listing_options("") == {}


def test_render_code_block():
    """A fenced block wraps code with the language tag."""
    assert render_code_block("print(1)", "python") == "```python\nprint(1)\n```"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_preprocess.py -v`
Expected: FAIL — `cannot import name 'slice_lines'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/tex2qmd/preprocess.py
"""Preprocess LaTeX chapter text before handing it to pandoc."""

from __future__ import annotations

from tex2qmd.listings import language_for_style


def slice_lines(text: str, firstline: int | None, lastline: int | None) -> str:
    """Return lines [firstline, lastline] (1-indexed, inclusive) of text."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    start = (firstline - 1) if firstline else 0
    end = lastline if lastline else len(lines)
    return "\n".join(lines[start:end])


def parse_listing_options(opts: str) -> dict[str, str]:
    """Parse a lstlisting/lstinputlisting `[...]` option string into a dict."""
    result: dict[str, str] = {}
    for part in opts.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        result[key.strip()] = value.strip()
    return result


def render_code_block(code: str, language: str) -> str:
    """Wrap code in a fenced block tagged with language."""
    return f"```{language}\n{code.rstrip(chr(10))}\n```"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_preprocess.py -v`
Expected: PASS (6 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/preprocess.py tests/tex2qmd/test_preprocess.py
git commit -m "feat(tex2qmd): add line slicing and listing-option parsing"
```

---

### Task 3: Extract listings to placeholder tokens

Both `\lstinputlisting{...}` and inline `\begin{lstlisting}...\end{lstlisting}` are replaced with unique plain-text tokens that survive pandoc untouched. The fenced code blocks are collected for reinsertion in postprocess.

**Files:**
- Modify: `scripts/tex2qmd/preprocess.py`
- Test: `tests/tex2qmd/test_extract_listings.py`

**Interfaces:**
- Consumes: `slice_lines`, `parse_listing_options`, `render_code_block`, `language_for_style`.
- Produces: `extract_listings(tex: str, base_dir: Path) -> tuple[str, list[str]]` — returns `(tex_with_tokens, code_blocks)` where each token has the form `TEX2QMDCODEBLOCK{i}` on its own paragraph, and `code_blocks[i]` is the fenced block. `base_dir` is the `latex/` directory used to resolve `\lstinputlisting` paths.

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_extract_listings.py
from pathlib import Path
from tex2qmd.preprocess import extract_listings


def test_extract_inline_listing(tmp_path):
    """Inline lstlisting becomes a token plus a python fenced block."""
    tex = "Before.\n\\begin{lstlisting}[style=Pythonshort]\nprint(1)\n\\end{lstlisting}\nAfter."
    out, blocks = extract_listings(tex, tmp_path)
    assert "TEX2QMDCODEBLOCK0" in out
    assert "\\begin{lstlisting}" not in out
    assert blocks == ["```python\nprint(1)\n```"]


def test_extract_input_listing_with_range(tmp_path):
    """lstinputlisting inlines the referenced file sliced by firstline/lastline."""
    src = tmp_path / "sample.py"
    src.write_text("l1\nl2\nl3\nl4\nl5\n")
    tex = "\\lstinputlisting[style=Python, firstline=2, lastline=4]{sample.py}"
    out, blocks = extract_listings(tex, tmp_path)
    assert out.strip() == "TEX2QMDCODEBLOCK0"
    assert blocks == ["```python\nl2\nl3\nl4\n```"]


def test_extract_multiple_listings_indexed(tmp_path):
    """Multiple listings get sequential tokens."""
    tex = (
        "\\begin{lstlisting}[style=shellshort]\n$ ls\n\\end{lstlisting}\n"
        "\\begin{lstlisting}[style=Pythonshort]\nx = 1\n\\end{lstlisting}"
    )
    out, blocks = extract_listings(tex, tmp_path)
    assert "TEX2QMDCODEBLOCK0" in out and "TEX2QMDCODEBLOCK1" in out
    assert blocks[0] == "```bash\n$ ls\n```"
    assert blocks[1] == "```python\nx = 1\n```"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_extract_listings.py -v`
Expected: FAIL — `cannot import name 'extract_listings'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/tex2qmd/preprocess.py`:

```python
import re
from pathlib import Path

_INPUT_LISTING = re.compile(
    r"\\lstinputlisting(?:\[(?P<opts>[^\]]*)\])?\{(?P<path>[^}]*)\}"
)
_INLINE_LISTING = re.compile(
    r"\\begin\{lstlisting\}(?:\[(?P<opts>[^\]]*)\])?\n?(?P<body>.*?)\\end\{lstlisting\}",
    re.DOTALL,
)


def _token(index: int) -> str:
    """Return the placeholder token for the given listing index."""
    return f"\n\nTEX2QMDCODEBLOCK{index}\n\n"


def extract_listings(tex: str, base_dir: Path) -> tuple[str, list[str]]:
    """Replace all listings with tokens; return (text, fenced code blocks)."""
    blocks: list[str] = []

    def _input_repl(match: re.Match[str]) -> str:
        opts = parse_listing_options(match.group("opts") or "")
        language = language_for_style(opts.get("style", ""))
        path = (base_dir / match.group("path")).resolve()
        text = path.read_text()
        first = int(opts["firstline"]) if "firstline" in opts else None
        last = int(opts["lastline"]) if "lastline" in opts else None
        code = slice_lines(text, first, last)
        blocks.append(render_code_block(code, language))
        return _token(len(blocks) - 1)

    def _inline_repl(match: re.Match[str]) -> str:
        opts = parse_listing_options(match.group("opts") or "")
        language = language_for_style(opts.get("style", ""))
        code = match.group("body").rstrip("\n")
        blocks.append(render_code_block(code, language))
        return _token(len(blocks) - 1)

    tex = _INPUT_LISTING.sub(_input_repl, tex)
    tex = _INLINE_LISTING.sub(_inline_repl, tex)
    return tex, blocks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_extract_listings.py -v`
Expected: PASS (3 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/preprocess.py tests/tex2qmd/test_extract_listings.py
git commit -m "feat(tex2qmd): extract listings to placeholder tokens"
```

---

### Task 4: Strip `\index` entries

**Files:**
- Modify: `scripts/tex2qmd/preprocess.py`
- Test: `tests/tex2qmd/test_strip_index.py`

**Interfaces:**
- Produces: `strip_index(tex: str) -> str` — removes all `\index{...}` commands (including `!` subentries and `@` sort keys) while leaving surrounding prose byte-for-byte otherwise unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_strip_index.py
from tex2qmd.preprocess import strip_index


def test_strip_simple_index():
    """A trailing \\index command is removed, prose preserved."""
    tex = "Tests matter.\\index{testing} They do."
    assert strip_index(tex) == "Tests matter. They do."


def test_strip_subentry_index():
    """Subentry index (with !) is removed."""
    tex = "unit tests\\index{testing!unit} are good."
    assert strip_index(tex) == "unit tests are good."


def test_strip_multiple_index():
    """Multiple index commands on one line are all removed."""
    tex = "a\\index{x}b\\index{y}c"
    assert strip_index(tex) == "abc"
```

Note on `test_strip_simple_index`: the removal must not leave a double space. The implementation removes exactly the `\index{...}` token; the input `"matter.\index{testing} They"` becomes `"matter. They"` (single space) because there is no space before `\index`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_strip_index.py -v`
Expected: FAIL — `cannot import name 'strip_index'`.

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/tex2qmd/preprocess.py`:

```python
_INDEX = re.compile(r"\\index\{[^{}]*\}")


def strip_index(tex: str) -> str:
    """Remove all \\index{...} commands from the text."""
    return _INDEX.sub("", tex)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_strip_index.py -v`
Expected: PASS (3 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/preprocess.py tests/tex2qmd/test_strip_index.py
git commit -m "feat(tex2qmd): strip index commands"
```

---

### Task 5: Figure conversion (PDF→SVG) and `\includegraphics` rewrite

**Files:**
- Create: `scripts/tex2qmd/figures.py`
- Test: `tests/tex2qmd/test_figures.py`

**Interfaces:**
- Produces:
  - `latex_width_to_percent(width: str) -> str | None` — converts `0.625\linewidth` → `62%`; returns `None` for widths it cannot interpret.
  - `svg_name(pdf_relpath: str) -> str` — maps `figures/07/foo.pdf` → `figures/07/foo.svg`.
  - `rewrite_includegraphics(tex: str) -> tuple[str, list[str]]` — rewrites every `\includegraphics` PDF path to its `.svg` sibling and appends a Quarto width attribute; returns `(tex, referenced_pdf_relpaths)`.
  - `convert_pdf_to_svg(pdf_path: Path, out_svg: Path) -> None` — runs `pdftocairo -svg`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_figures.py
from tex2qmd.figures import latex_width_to_percent, svg_name, rewrite_includegraphics


def test_latex_width_to_percent():
    """A fractional linewidth becomes an integer percentage."""
    assert latex_width_to_percent("0.625\\linewidth") == "62%"
    assert latex_width_to_percent("0.5\\textwidth") == "50%"


def test_latex_width_to_percent_unparseable():
    """A width with no recognizable fraction returns None."""
    assert latex_width_to_percent("3cm") is None


def test_svg_name():
    """A PDF figure path maps to its .svg sibling."""
    assert svg_name("figures/07/lifecycle.pdf") == "figures/07/lifecycle.svg"


def test_rewrite_includegraphics():
    """includegraphics is rewritten to svg with a width attribute; pdf collected."""
    tex = "\\includegraphics[width=0.5\\linewidth]{figures/02/LabNotebook.pdf}"
    out, pdfs = rewrite_includegraphics(tex)
    assert out == "\\includegraphics[width=50%]{figures/02/LabNotebook.svg}"
    assert pdfs == ["figures/02/LabNotebook.pdf"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_figures.py -v`
Expected: FAIL — `No module named 'tex2qmd.figures'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/tex2qmd/figures.py
"""Convert figure PDFs to SVG and rewrite \\includegraphics references."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_FRACTION = re.compile(r"(?P<frac>[0-9.]+)\\(?:line|text|column)width")
_INCLUDE = re.compile(
    r"\\includegraphics(?:\[(?P<opts>[^\]]*)\])?\{(?P<path>[^}]*)\}"
)


def latex_width_to_percent(width: str) -> str | None:
    """Convert a fractional LaTeX width to an integer percentage string."""
    match = _FRACTION.search(width)
    if not match:
        return None
    return f"{int(float(match.group('frac')) * 100)}%"


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
        percent = latex_width_to_percent(match.group("opts") or "")
        opts = f"[width={percent}]" if percent else ""
        return f"\\includegraphics{opts}{{{svg_name(path)}}}"

    return _INCLUDE.sub(_repl, tex), pdfs


def convert_pdf_to_svg(pdf_path: Path, out_svg: Path) -> None:
    """Convert a single PDF figure to SVG using pdftocairo."""
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftocairo", "-svg", str(pdf_path), str(out_svg)],
        check=True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_figures.py -v`
Expected: PASS (4 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/figures.py tests/tex2qmd/test_figures.py
git commit -m "feat(tex2qmd): convert figures to svg and rewrite references"
```

---

### Task 6: Chapter order + `_quarto.yml` generation

**Files:**
- Create: `scripts/tex2qmd/project.py`
- Test: `tests/tex2qmd/test_project.py`

**Interfaces:**
- Produces:
  - `parse_chapter_order(book_tex: str) -> list[str]` — returns `\include{...}` targets in document order (e.g. `["preface", "book-introduction", ...]`).
  - `qmd_name(include_target: str) -> str` — `"book-testing"` → `"book-testing.qmd"`; the first entry (`preface`) is special-cased by the caller to `index.qmd`.
  - `render_quarto_yml(chapter_files: list[str], title: str, author: str) -> str` — returns the `_quarto.yml` contents with `project: type: book`, chapter list, `bibliography: references.bib`, `csl: cambridge.csl`, HTML theme, and search enabled.

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_project.py
from tex2qmd.project import parse_chapter_order, qmd_name, render_quarto_yml


def test_parse_chapter_order():
    """Include targets are returned in document order, comments ignored."""
    book_tex = (
        "\\begin{document}\n"
        "\\include{preface}\n"
        "% \\include{skipped}\n"
        "\\include{book-introduction}\n"
        "\\include{book-testing}\n"
    )
    assert parse_chapter_order(book_tex) == [
        "preface",
        "book-introduction",
        "book-testing",
    ]


def test_qmd_name():
    """An include target maps to a .qmd filename."""
    assert qmd_name("book-testing") == "book-testing.qmd"


def test_render_quarto_yml_contains_chapters_and_bib():
    """The rendered config lists chapters and wires up bibliography + csl."""
    yml = render_quarto_yml(
        ["index.qmd", "book-testing.qmd"], "Better Code, Better Science", "Russell A. Poldrack"
    )
    assert "type: book" in yml
    assert "index.qmd" in yml
    assert "book-testing.qmd" in yml
    assert "bibliography: references.bib" in yml
    assert "csl: cambridge.csl" in yml
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_project.py -v`
Expected: FAIL — `No module named 'tex2qmd.project'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/tex2qmd/project.py
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


def render_quarto_yml(chapter_files: list[str], title: str, author: str) -> str:
    """Render the _quarto.yml contents for the HTML book."""
    chapter_lines = "\n".join(f"    - {name}" for name in chapter_files)
    return f"""project:
  type: book

book:
  title: "{title}"
  author: "{author}"
  search: true
  chapters:
{chapter_lines}

bibliography: references.bib
csl: cambridge.csl

format:
  html:
    theme: cosmo
    toc: true
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_project.py -v`
Expected: PASS (3 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/project.py tests/tex2qmd/test_project.py
git commit -m "feat(tex2qmd): parse chapter order and render _quarto.yml"
```

---

### Task 7: Pandoc bridge

**Files:**
- Create: `scripts/tex2qmd/pandoc_bridge.py`
- Test: `tests/tex2qmd/test_pandoc_bridge.py`

**Interfaces:**
- Produces: `run_pandoc(latex: str) -> str` — runs `pandoc --from latex --to markdown --wrap=preserve` on the input string via stdin/stdout and returns markdown. Requires the `pandoc` binary (integration test).

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_pandoc_bridge.py
import shutil
import pytest
from tex2qmd.pandoc_bridge import run_pandoc

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_run_pandoc_converts_section_and_citation():
    """Pandoc converts a LaTeX section and \\citep to markdown."""
    latex = "\\section{Intro}\nSee \\citep{smith2020} for details."
    md = run_pandoc(latex)
    assert "# Intro" in md
    assert "[@smith2020]" in md


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_run_pandoc_preserves_placeholder_token():
    """A code placeholder token survives pandoc as plain text."""
    latex = "Before.\n\nTEX2QMDCODEBLOCK0\n\nAfter."
    md = run_pandoc(latex)
    assert "TEX2QMDCODEBLOCK0" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_pandoc_bridge.py -v`
Expected: FAIL — `No module named 'tex2qmd.pandoc_bridge'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/tex2qmd/pandoc_bridge.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_pandoc_bridge.py -v`
Expected: PASS (2 cases). If `test_run_pandoc_preserves_placeholder_token` shows the token wrapped/escaped, adjust the token format in Task 3 (`_token`) to a value pandoc passes through verbatim and re-run both tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/pandoc_bridge.py tests/tex2qmd/test_pandoc_bridge.py
git commit -m "feat(tex2qmd): add pandoc bridge"
```

---

### Task 8: Postprocess — reinsert code blocks + normalize

**Files:**
- Create: `scripts/tex2qmd/postprocess.py`
- Test: `tests/tex2qmd/test_postprocess.py`

**Interfaces:**
- Produces: `reinsert_code_blocks(md: str, blocks: list[str]) -> str` — replaces each `TEX2QMDCODEBLOCK{i}` token (however pandoc emitted it on its own line) with `blocks[i]`. Raises `ValueError` if a token has no matching block or a block is never referenced.

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_postprocess.py
import pytest
from tex2qmd.postprocess import reinsert_code_blocks


def test_reinsert_single_block():
    """A token line is replaced by its fenced code block."""
    md = "Before.\n\nTEX2QMDCODEBLOCK0\n\nAfter."
    blocks = ["```python\nprint(1)\n```"]
    out = reinsert_code_blocks(md, blocks)
    assert "```python\nprint(1)\n```" in out
    assert "TEX2QMDCODEBLOCK0" not in out


def test_reinsert_multiple_blocks_in_order():
    """Multiple tokens map to their indexed blocks."""
    md = "TEX2QMDCODEBLOCK0\n\ntext\n\nTEX2QMDCODEBLOCK1"
    blocks = ["```bash\nls\n```", "```python\nx=1\n```"]
    out = reinsert_code_blocks(md, blocks)
    assert out.index("ls") < out.index("x=1")


def test_reinsert_missing_block_raises():
    """A token with no matching block is an error."""
    md = "TEX2QMDCODEBLOCK5"
    with pytest.raises(ValueError):
        reinsert_code_blocks(md, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_postprocess.py -v`
Expected: FAIL — `No module named 'tex2qmd.postprocess'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/tex2qmd/postprocess.py
"""Reinsert extracted code blocks into pandoc's markdown output."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"TEX2QMDCODEBLOCK(?P<index>\d+)")


def reinsert_code_blocks(md: str, blocks: list[str]) -> str:
    """Replace each code placeholder token with its fenced code block."""
    seen: set[int] = set()

    def _repl(match: re.Match[str]) -> str:
        index = int(match.group("index"))
        if index >= len(blocks):
            raise ValueError(f"No code block for token index {index}")
        seen.add(index)
        return blocks[index]

    out = _TOKEN.sub(_repl, md)
    if len(seen) != len(blocks):
        missing = set(range(len(blocks))) - seen
        raise ValueError(f"Code blocks never referenced: {sorted(missing)}")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_postprocess.py -v`
Expected: PASS (3 cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/postprocess.py tests/tex2qmd/test_postprocess.py
git commit -m "feat(tex2qmd): reinsert code blocks in postprocess"
```

---

### Task 9: Single-chapter conversion orchestration

Wire the pieces into a function that converts one chapter's LaTeX text to a `.qmd` string and collects figure PDFs to convert.

**Files:**
- Create: `scripts/tex2qmd/convert.py`
- Test: `tests/tex2qmd/test_convert.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `convert_chapter(tex: str, base_dir: Path) -> tuple[str, list[str]]` — returns `(qmd_text, pdf_figure_relpaths)`. Pipeline order: `strip_index` → `rewrite_includegraphics` (collect pdfs) → `extract_listings` (collect blocks) → `run_pandoc` → `reinsert_code_blocks`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_convert.py
import shutil
import pytest
from tex2qmd.convert import convert_chapter

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_convert_chapter_end_to_end(tmp_path):
    """A small chapter converts: prose to md, listing to fence, index gone, fig collected."""
    (tmp_path / "sample.py").write_text("print('hi')\n")
    tex = (
        "\\section{Demo}\\index{demo}\n"
        "Text with \\citep{ref1}.\n"
        "\\includegraphics[width=0.5\\linewidth]{figures/01/x.pdf}\n"
        "\\lstinputlisting[style=Python]{sample.py}\n"
    )
    qmd, pdfs = convert_chapter(tex, tmp_path)
    assert "# Demo" in qmd
    assert "[@ref1]" in qmd
    assert "\\index" not in qmd
    assert "```python\nprint('hi')\n```" in qmd
    assert pdfs == ["figures/01/x.pdf"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_convert.py -v`
Expected: FAIL — `cannot import name 'convert_chapter'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/tex2qmd/convert.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_convert.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/convert.py tests/tex2qmd/test_convert.py
git commit -m "feat(tex2qmd): orchestrate single-chapter conversion"
```

---

### Task 10: Whole-book build CLI

Add a `main()` CLI that builds the entire `web/` project: reads `book.tex`, converts each chapter, writes `.qmd` files, converts figures, writes `_quarto.yml`, and copies `references.bib` + `cambridge.csl`.

**Files:**
- Modify: `scripts/tex2qmd/convert.py`
- Create: `scripts/tex2qmd/cambridge.csl` (author-date CSL; fetch the CSL project's `cambridge-university-press-author-date.csl`, committed as `cambridge.csl`)
- Test: `tests/tex2qmd/test_build.py`

**Interfaces:**
- Consumes: `parse_chapter_order`, `qmd_name`, `render_quarto_yml`, `convert_chapter`, `convert_pdf_to_svg`.
- Produces: `build_book(latex_dir: Path, out_dir: Path) -> None` and a `main()` argparse entry point invoked as `uv run python scripts/tex2qmd/convert.py [--latex-dir latex] [--out web]`. `build_book` writes: `out_dir/index.qmd` (from `preface`), `out_dir/<chapter>.qmd` for each remaining include, `out_dir/figures/**.svg`, `out_dir/_quarto.yml`, `out_dir/references.bib`, `out_dir/cambridge.csl`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tex2qmd/test_build.py
import shutil
import pytest
from pathlib import Path
from tex2qmd.convert import build_book

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
def test_build_book_writes_project(tmp_path):
    """build_book emits _quarto.yml, index.qmd, chapter qmd, and copies bib."""
    latex = tmp_path / "latex"
    latex.mkdir()
    (latex / "book.tex").write_text(
        "\\begin{document}\n\\include{preface}\n\\include{chap1}\n\\end{document}\n"
    )
    (latex / "preface.tex").write_text("\\chapter*{Preface}\nHello.\n")
    (latex / "chap1.tex").write_text("\\chapter{One}\nBody \\citep{k}.\n")
    (latex / "references.bib").write_text("@book{k, title={T}, year={2020}}\n")
    out = tmp_path / "web"
    build_book(latex, out)
    assert (out / "_quarto.yml").exists()
    assert (out / "index.qmd").exists()
    assert (out / "chap1.qmd").exists()
    assert (out / "references.bib").exists()
    assert (out / "cambridge.csl").exists()
    assert "type: book" in (out / "_quarto.yml").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_build.py -v`
Expected: FAIL — `cannot import name 'build_book'`.

- [ ] **Step 3: Write minimal implementation**

First obtain the CSL file (one time):

```bash
curl -sL https://raw.githubusercontent.com/citation-style-language/styles/master/cambridge-university-press-author-date.csl \
  -o scripts/tex2qmd/cambridge.csl
```

Add to `scripts/tex2qmd/convert.py`:

```python
import argparse
import shutil

from tex2qmd.figures import convert_pdf_to_svg, svg_name
from tex2qmd.project import parse_chapter_order, qmd_name, render_quarto_yml

TITLE = "Better Code, Better Science"
AUTHOR = "Russell A. Poldrack"


def build_book(latex_dir: Path, out_dir: Path) -> None:
    """Convert every chapter in book.tex into a Quarto project under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    book_tex = (latex_dir / "book.tex").read_text()
    targets = parse_chapter_order(book_tex)

    chapter_files: list[str] = []
    for i, target in enumerate(targets):
        tex = (latex_dir / f"{target}.tex").read_text()
        qmd, pdfs = convert_chapter(tex, latex_dir)
        out_name = "index.qmd" if i == 0 else qmd_name(target)
        (out_dir / out_name).write_text(qmd)
        chapter_files.append(out_name)
        for pdf in pdfs:
            convert_pdf_to_svg(latex_dir / pdf, out_dir / svg_name(pdf))

    (out_dir / "_quarto.yml").write_text(
        render_quarto_yml(chapter_files, TITLE, AUTHOR)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/test_build.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd/convert.py scripts/tex2qmd/cambridge.csl tests/tex2qmd/test_build.py
git commit -m "feat(tex2qmd): build full Quarto project from book.tex"
```

---

### Task 11: Real testing-chapter conversion + gitignore + Makefile targets

Validate against the real `book-testing.tex`, wire up the developer workflow, and ignore generated output. This is a verification + integration task; no new pure logic.

**Files:**
- Modify: `.gitignore` (repo root)
- Modify: `latex/Makefile`

- [ ] **Step 1: Ignore generated output**

Append to the repo-root `.gitignore`:

```
# Generated Quarto HTML book (regenerated from LaTeX by scripts/tex2qmd)
/web/
```

- [ ] **Step 2: Regenerate listing outputs, then build the web project**

Run:

```bash
cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode
uv run python scripts/build_listings.py verify testing
uv run python scripts/tex2qmd/convert.py --latex-dir latex --out web
```

Expected: `web/` contains `index.qmd`, `book-testing.qmd` (and the other chapter `.qmd` files), `figures/**/*.svg`, `_quarto.yml`, `references.bib`, `cambridge.csl`. The command exits 0. If a chapter other than testing raises during conversion, note it — Task 12 iterates on those; for this task confirm `book-testing.qmd` is produced and well-formed.

- [ ] **Step 3: Render and eyeball the testing chapter**

Run:

```bash
cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode/web
quarto render book-testing.qmd --to html
```

Expected: renders without error to `web/_book/` (or `book-testing.html`). Open it and verify against the PDF: section structure, code listings show with correct syntax highlighting, figures appear, citations render as author-date, no stray `\index`/LaTeX commands.

- [ ] **Step 4: Add Makefile targets**

Add to `latex/Makefile`:

```makefile
# Regenerate the Quarto HTML book from the LaTeX source
quarto: verify-listings
	cd .. && uv run python scripts/tex2qmd/convert.py --latex-dir latex --out web
	cd ../web && quarto render

# Live preview of the Quarto book during development
quarto-serve:
	cd ../web && quarto preview
```

Add `quarto quarto-serve` to the `.PHONY` line.

- [ ] **Step 5: Commit**

```bash
cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode
git add .gitignore latex/Makefile
git commit -m "build(tex2qmd): add quarto make targets and ignore web output"
```

---

### Task 12: Generalize across all chapters

Run the full build, fix per-chapter conversion edge cases (unusual environments, macros pandoc chokes on), and confirm the whole book renders.

**Files:**
- Modify: `scripts/tex2qmd/preprocess.py` / `postprocess.py` as needed for edge cases discovered.
- Test: add regression tests to the relevant `tests/tex2qmd/test_*.py` for each edge case fixed.

- [ ] **Step 1: Full build**

Run:

```bash
cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode
uv run python scripts/tex2qmd/convert.py --latex-dir latex --out web
cd web && quarto render
```

Expected: all 12 chapters + preface render. Capture any chapter that errors.

- [ ] **Step 2: For each failure, add a failing regression test first**

For a discovered construct (e.g. a `framed` block that should become a Quarto callout, or a `tabular` that pandoc mangles), add a unit test to the appropriate `test_*.py` reproducing the minimal input/expected output, run it to confirm RED, then implement the fix in `preprocess.py`/`postprocess.py`, run to GREEN. One construct = one test + one fix + one commit.

Example skeleton for a `framed`→callout fix:

```python
# in tests/tex2qmd/test_postprocess.py
def test_framed_becomes_callout():
    """A pandoc-emitted framed block is normalized to a Quarto note callout."""
    md = "::: {.framed}\nImportant.\n:::"
    assert "::: {.callout-note}" in normalize_callouts(md)
```

- [ ] **Step 3: Re-run the full build until clean**

Run the Step 1 commands again after each fix. Expected end state: `cd web && quarto render` completes with all chapters, no errors.

- [ ] **Step 4: Run the whole tex2qmd test suite**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run pytest tests/tex2qmd/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/tex2qmd tests/tex2qmd
git commit -m "feat(tex2qmd): handle per-chapter conversion edge cases"
```

---

### Task 13: GitHub Actions → GitHub Pages deploy

**Files:**
- Create: `.github/workflows/deploy-quarto.yml`

**Interfaces:** none (CI only).

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/deploy-quarto.yml
name: Deploy Quarto book

on:
  push:
    branches: [main, latex-integration]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install system tools (poppler for pdftocairo)
        run: sudo apt-get update && sudo apt-get install -y poppler-utils

      - uses: astral-sh/setup-uv@v5

      - uses: quarto-dev/quarto-actions/setup@v2

      - name: Install Python deps
        run: uv sync

      - name: Regenerate listing outputs
        run: uv run python scripts/build_listings.py verify testing

      - name: Convert LaTeX to Quarto
        run: uv run python scripts/tex2qmd/convert.py --latex-dir latex --out web

      - name: Render Quarto book
        run: cd web && quarto render

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: web/_book

  deploy:
    needs: build-deploy
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Validate YAML locally**

Run: `cd /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode && uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/deploy-quarto.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/deploy-quarto.yml
git commit -m "ci(tex2qmd): deploy Quarto book to GitHub Pages"
```

- [ ] **Step 4: Enable Pages (manual, one-time)**

In the GitHub repo settings → Pages → Source: "GitHub Actions". Then push and confirm the workflow publishes the site. (Requires the author; note it in the handoff.)

---

## Notes for the implementer

- The placeholder-token strategy (Task 3/7/8) is the crux: pandoc handles prose/math/citations/cross-refs; the pipeline owns code-fence languages. If pandoc ever mangles a token, change `_token()` in `preprocess.py` to a form pandoc preserves (verified by `test_run_pandoc_preserves_placeholder_token`) and keep the same regex in `postprocess.py`.
- Chapter `.qmd` files, `web/`, and rendered HTML are all regenerable — never hand-edit them. Fix the pipeline instead.
- Integration tests are marked `@pytest.mark.integration` and skip when `pandoc` is absent, so the pure-unit suite stays fast and hermetic.
