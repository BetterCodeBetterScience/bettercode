# Reproducible LaTeX Listings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-27-latex-reproducible-listings-design.md`

**Goal:** Replace hand-curated `lstlisting` blocks in `latex/book-testing.tex` with build-time `\lstinputlisting` references to canonical Python files and to outputs captured by actually running pytest / a Python REPL.

**Architecture:** A per-chapter YAML manifest declares (command-or-driver → output_file → normalizers). A small Python CLI (`scripts/build_listings.py`) reads the manifest, runs each entry, applies named string-normalizing filters, and either writes the result (`build`) or diffs against the committed file (`verify`). LaTeX never embeds dynamic content — it `\lstinputlisting`s text files. Verification is wired into the PDF build so drift hard-fails.

**Tech Stack:** Python 3.12, `code.InteractiveConsole` (stdlib) for REPL transcript rendering, PyYAML for manifest parsing, pytest for tests, `uv run` for invocations, latexmk + xelatex for PDF.

**Conventions used throughout this plan:**
- All shell commands assume CWD = repo root unless stated otherwise.
- All Python commands use `uv run`.
- Every code-change task ends with a commit step.
- All `__init__.py` files must be **empty** (per project CLAUDE.md).
- Tests use functions, not classes (per project CLAUDE.md). Use `@pytest.fixture` for shared resources.
- Docstrings are NumPy-style.

**File structure being built:**
```
scripts/
  build_listings.py            # CLI entry point (build/verify modes)
  normalize_output.py          # Pure str->str normalizer functions, registered by name
  manifest.py                  # Dataclass + YAML loader with validation
  repl_console.py              # Session class wrapping code.InteractiveConsole
  repl_drivers/testing/        # One driver per REPL block in the chapter
    <id>.py

src/bettercode/testing/
  __init__.py                  # empty
  <concept>_v<N>.py            # one file per pedagogical version

tests/test_listings_infra/
  __init__.py                  # empty
  conftest.py                  # adds scripts/ to sys.path
  test_normalize_output.py
  test_manifest.py
  test_repl_console.py
  test_build_listings.py

tests/testing_chapter/
  __init__.py                  # empty
  conftest.py                  # if shared fixtures needed
  test_<concept>.py            # tests featured in the chapter

latex/manifests/testing.yaml
latex/generated/testing/<id>.txt
latex/CONVERSION_NOTES.md      # running log; raw material for future skill
```

---

## Phase A — Infrastructure (TDD)

These tasks build the listing tooling. They do not depend on the testing chapter content and can be completed independently.

### Task 1: Add PyYAML dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add PyYAML to dependencies**

Open `pyproject.toml`. Inside `[project] dependencies = [ ... ]`, add a line for PyYAML (alphabetically, after another item that starts with `p`):

```toml
    "pyyaml>=6.0",
```

- [ ] **Step 2: Sync and verify import**

```bash
uv sync
uv run python -c "import yaml; print(yaml.__version__)"
```
Expected: prints a version string ≥ 6.0.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pyyaml for listings manifest parsing"
```

---

### Task 2: Set up listings-infra test scaffolding

**Files:**
- Create: `tests/test_listings_infra/__init__.py` (empty)
- Create: `tests/test_listings_infra/conftest.py`

- [ ] **Step 1: Create empty `__init__.py`**

Create `tests/test_listings_infra/__init__.py` as an empty file.

- [ ] **Step 2: Create `conftest.py`**

Create `tests/test_listings_infra/conftest.py` with:

```python
"""Test config for the listings-build infrastructure.

Adds the `scripts/` directory to sys.path so tests can import the
script modules directly (they are not installed as a package).
"""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
```

- [ ] **Step 3: Verify pytest picks it up**

```bash
uv run pytest tests/test_listings_infra/ -v
```
Expected: "no tests ran" (exit 5) — confirms collection works without errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_listings_infra/__init__.py tests/test_listings_infra/conftest.py
git commit -m "test: scaffold tests/test_listings_infra/ for build tooling"
```

---

### Task 3: Implement `normalize_output.py` (pytest_timings normalizer)

**Files:**
- Create: `tests/test_listings_infra/test_normalize_output.py`
- Create: `scripts/normalize_output.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_listings_infra/test_normalize_output.py`:

```python
"""Tests for output normalizers."""
from normalize_output import apply_normalizers, NORMALIZERS


def test_pytest_timings_replaces_decimal_seconds():
    text = "===== 1 passed in 0.10s ====="
    result = NORMALIZERS["pytest_timings"](text)
    assert result == "===== 1 passed in X.XXs ====="


def test_pytest_timings_handles_multiple_occurrences():
    text = "1 failed, 2 passed in 1.23s\nshort test summary: 0.45s\n"
    result = NORMALIZERS["pytest_timings"](text)
    assert "X.XXs" in result
    assert "1.23s" not in result
    assert "0.45s" not in result


def test_pytest_timings_leaves_non_timing_text_untouched():
    text = "tests/test_foo.py::test_bar PASSED"
    assert NORMALIZERS["pytest_timings"](text) == text
```

- [ ] **Step 2: Run test to confirm RED**

```bash
uv run pytest tests/test_listings_infra/test_normalize_output.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'normalize_output'`.

- [ ] **Step 3: Implement minimally**

Create `scripts/normalize_output.py`:

```python
"""Named text normalizers applied to captured command output.

Each normalizer is a pure `str -> str` function registered in NORMALIZERS
by name. Manifests reference normalizers by these names.
"""
from __future__ import annotations

import re
from typing import Callable

NORMALIZERS: dict[str, Callable[[str], str]] = {}


def register(name: str) -> Callable[[Callable[[str], str]], Callable[[str], str]]:
    """Decorator: register a normalizer function under `name`."""
    def _wrap(fn: Callable[[str], str]) -> Callable[[str], str]:
        NORMALIZERS[name] = fn
        return fn
    return _wrap


@register("pytest_timings")
def pytest_timings(text: str) -> str:
    """Replace pytest timing strings like '0.10s' with 'X.XXs'."""
    return re.sub(r"\b\d+\.\d+s\b", "X.XXs", text)


def apply_normalizers(text: str, names: list[str]) -> str:
    """Apply each named normalizer in order. Unknown names raise KeyError."""
    for name in names:
        text = NORMALIZERS[name](text)
    return text
```

- [ ] **Step 4: Run test to verify GREEN**

```bash
uv run pytest tests/test_listings_infra/test_normalize_output.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/normalize_output.py tests/test_listings_infra/test_normalize_output.py
git commit -m "feat(listings): add pytest_timings normalizer with registry"
```

---

### Task 4: Add remaining normalizers (memory_addresses, file_paths, temp_paths, terminal_width)

**Files:**
- Modify: `tests/test_listings_infra/test_normalize_output.py`
- Modify: `scripts/normalize_output.py`

- [ ] **Step 1: Add failing tests for memory_addresses**

Append to `tests/test_listings_infra/test_normalize_output.py`:

```python
def test_memory_addresses_collapses_hex_pointers():
    text = "<function allclose at 0x101403370>"
    result = NORMALIZERS["memory_addresses"](text)
    assert result == "<function allclose at 0xADDR>"


def test_memory_addresses_only_hits_0x_prefix():
    text = "expected 100, got 200"
    assert NORMALIZERS["memory_addresses"](text) == text
```

- [ ] **Step 2: Implement memory_addresses**

In `scripts/normalize_output.py`, append:

```python
@register("memory_addresses")
def memory_addresses(text: str) -> str:
    """Replace hex memory addresses like '0x101403370' with '0xADDR'."""
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
```

Run: `uv run pytest tests/test_listings_infra/test_normalize_output.py -v`
Expected: all tests pass (5 total).

- [ ] **Step 3: Add failing tests for file_paths**

Append:

```python
def test_file_paths_strips_repo_absolute_to_relative(tmp_path, monkeypatch):
    """Absolute paths under the repo root are rewritten to repo-relative."""
    monkeypatch.chdir(tmp_path)
    repo_root = str(tmp_path)
    text = f"  File \"{repo_root}/src/bettercode/foo.py\", line 12, in foo"
    result = NORMALIZERS["file_paths"](text)
    assert result == "  File \"src/bettercode/foo.py\", line 12, in foo"


def test_file_paths_leaves_non_repo_paths_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = "  File \"/usr/lib/python3.12/site-packages/x.py\", line 1, in y"
    assert NORMALIZERS["file_paths"](text) == text
```

- [ ] **Step 4: Implement file_paths**

Append to `scripts/normalize_output.py`:

```python
import os


@register("file_paths")
def file_paths(text: str) -> str:
    """Rewrite absolute paths under the current working directory to be repo-relative."""
    cwd = os.getcwd()
    # Use a regex that matches quoted or unquoted absolute paths under cwd.
    # cwd may contain regex metachars; escape it.
    pattern = re.compile(re.escape(cwd) + r"/")
    return pattern.sub("", text)
```

Run: `uv run pytest tests/test_listings_infra/test_normalize_output.py -v`
Expected: 7 tests pass.

- [ ] **Step 5: Add failing tests for temp_paths**

Append:

```python
def test_temp_paths_normalizes_macos_temp():
    text = "wrote /var/folders/aa/bb/T/tmpXYZ.txt"
    result = NORMALIZERS["temp_paths"](text)
    assert result == "wrote /tmp/PATH.txt"


def test_temp_paths_normalizes_generic_tmp():
    text = "opened /tmp/pytest-of-user/pytest-12/abc.txt"
    result = NORMALIZERS["temp_paths"](text)
    assert "pytest-of-user" not in result
    assert "/tmp/PATH" in result
```

- [ ] **Step 6: Implement temp_paths**

Append:

```python
@register("temp_paths")
def temp_paths(text: str) -> str:
    """Collapse temp-directory paths to '/tmp/PATH'."""
    # macOS: /var/folders/.../T/...
    text = re.sub(r"/var/folders/[^\s\"']+", "/tmp/PATH", text)
    # Linux/pytest: /tmp/pytest-of-*/...
    text = re.sub(r"/tmp/pytest-[^\s\"']+", "/tmp/PATH", text)
    return text
```

Run: `uv run pytest tests/test_listings_infra/test_normalize_output.py -v`
Expected: 9 tests pass.

- [ ] **Step 7: Add failing tests for terminal_width**

Append:

```python
def test_terminal_width_truncates_long_rule_lines():
    text = "=" * 120 + "\nactual content\n" + "=" * 90
    result = NORMALIZERS["terminal_width"](text)
    lines = result.split("\n")
    assert lines[0] == "=" * 64
    assert lines[1] == "actual content"
    assert lines[2] == "=" * 64


def test_terminal_width_leaves_short_rules_alone():
    text = "==== short rule ====\n"
    assert NORMALIZERS["terminal_width"](text) == text
```

- [ ] **Step 8: Implement terminal_width**

Append:

```python
RULE_WIDTH = 64


@register("terminal_width")
def terminal_width(text: str) -> str:
    """Truncate lines made entirely of '=' characters to RULE_WIDTH columns.

    Pytest renders separator rules using the terminal width, which makes
    output unstable across environments. Normalize to a fixed width.
    """
    out_lines = []
    for line in text.split("\n"):
        if line and set(line) == {"="}:
            out_lines.append("=" * RULE_WIDTH)
        else:
            out_lines.append(line)
    return "\n".join(out_lines)
```

Run: `uv run pytest tests/test_listings_infra/test_normalize_output.py -v`
Expected: 11 tests pass.

- [ ] **Step 9: Add test for apply_normalizers ordering**

Append:

```python
def test_apply_normalizers_runs_in_order():
    text = "function at 0x123 ran in 0.5s"
    result = apply_normalizers(text, ["memory_addresses", "pytest_timings"])
    assert result == "function at 0xADDR ran in X.XXs"


def test_apply_normalizers_unknown_name_raises():
    import pytest
    with pytest.raises(KeyError):
        apply_normalizers("anything", ["does_not_exist"])
```

Run: `uv run pytest tests/test_listings_infra/test_normalize_output.py -v`
Expected: 13 tests pass.

- [ ] **Step 10: Commit**

```bash
git add scripts/normalize_output.py tests/test_listings_infra/test_normalize_output.py
git commit -m "feat(listings): add memory_addresses, file_paths, temp_paths, terminal_width normalizers"
```

---

### Task 5: Implement `manifest.py` (YAML loader + validation)

**Files:**
- Create: `tests/test_listings_infra/test_manifest.py`
- Create: `scripts/manifest.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_listings_infra/test_manifest.py`:

```python
"""Tests for the listings-manifest loader."""
from pathlib import Path

import pytest
from manifest import ManifestEntry, load_manifest


def test_load_manifest_parses_command_entry(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: pytest_pass
    description: "Passing pytest"
    command: ["pytest", "tests/foo.py"]
    output_file: "latex/generated/testing/pytest_pass.txt"
    expected_exit_code: 0
    normalize: [pytest_timings]
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    m = load_manifest(p)
    assert m.chapter == "testing"
    assert len(m.outputs) == 1
    entry = m.outputs[0]
    assert entry.id == "pytest_pass"
    assert entry.command == ["pytest", "tests/foo.py"]
    assert entry.driver is None
    assert entry.output_file == "latex/generated/testing/pytest_pass.txt"
    assert entry.expected_exit_code == 0
    assert entry.normalize == ["pytest_timings"]


def test_load_manifest_parses_driver_entry(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: repl_one
    description: "REPL one"
    driver: "scripts/repl_drivers/testing/one.py"
    output_file: "latex/generated/testing/repl_one.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    m = load_manifest(p)
    entry = m.outputs[0]
    assert entry.driver == "scripts/repl_drivers/testing/one.py"
    assert entry.command is None


def test_load_manifest_rejects_both_command_and_driver(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: bad
    description: "Bad"
    command: ["true"]
    driver: "scripts/x.py"
    output_file: "latex/generated/testing/bad.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="exactly one of"):
        load_manifest(p)


def test_load_manifest_rejects_neither_command_nor_driver(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: bad
    description: "Bad"
    output_file: "latex/generated/testing/bad.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="exactly one of"):
        load_manifest(p)


def test_load_manifest_rejects_duplicate_ids(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: same
    description: "A"
    command: ["true"]
    output_file: "latex/generated/testing/a.txt"
    expected_exit_code: 0
    normalize: []
  - id: same
    description: "B"
    command: ["true"]
    output_file: "latex/generated/testing/b.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)


def test_load_manifest_defaults_cwd_to_dot(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: x
    description: "X"
    command: ["true"]
    output_file: "latex/generated/testing/x.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    m = load_manifest(p)
    assert m.outputs[0].cwd == "."
```

- [ ] **Step 2: Run tests to confirm RED**

```bash
uv run pytest tests/test_listings_infra/test_manifest.py -v
```
Expected: `ModuleNotFoundError: No module named 'manifest'`.

- [ ] **Step 3: Implement `manifest.py`**

Create `scripts/manifest.py`:

```python
"""Listings manifest: dataclasses + YAML loader with validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ManifestEntry:
    """One output to be generated and verified."""
    id: str
    description: str
    output_file: str
    expected_exit_code: int
    normalize: list[str]
    command: Optional[list[str]] = None
    driver: Optional[str] = None
    cwd: str = "."

    def __post_init__(self) -> None:
        has_command = self.command is not None
        has_driver = self.driver is not None
        if has_command == has_driver:
            raise ValueError(
                f"entry {self.id!r}: exactly one of 'command' or 'driver' "
                f"must be set"
            )


@dataclass
class Manifest:
    """Parsed listings manifest for a single chapter."""
    chapter: str
    outputs: list[ManifestEntry] = field(default_factory=list)


def load_manifest(path: Path) -> Manifest:
    """Load and validate a chapter manifest from `path`.

    Raises
    ------
    ValueError
        If the manifest is missing required fields, has duplicate output
        ids, or an entry specifies both/neither of `command` and `driver`.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if "chapter" not in raw:
        raise ValueError(f"{path}: missing required field 'chapter'")
    if "outputs" not in raw:
        raise ValueError(f"{path}: missing required field 'outputs'")

    entries = [ManifestEntry(**item) for item in raw["outputs"]]

    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            raise ValueError(f"{path}: duplicate entry id {entry.id!r}")
        seen.add(entry.id)

    return Manifest(chapter=raw["chapter"], outputs=entries)
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
uv run pytest tests/test_listings_infra/test_manifest.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/manifest.py tests/test_listings_infra/test_manifest.py
git commit -m "feat(listings): add manifest dataclass + yaml loader with validation"
```

---

### Task 6: Implement `repl_console.py` (REPL Session)

**Files:**
- Create: `tests/test_listings_infra/test_repl_console.py`
- Create: `scripts/repl_console.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_listings_infra/test_repl_console.py`:

```python
"""Tests for the REPL Session helper."""
from repl_console import Session


def test_simple_expression_renders_with_prompt_and_result():
    s = Session()
    s.run("1 + 1")
    assert s.render() == ">>> 1 + 1\n2\n"


def test_assignment_produces_no_output():
    s = Session()
    s.run("x = 5")
    assert s.render() == ">>> x = 5\n"


def test_state_persists_across_run_calls():
    s = Session()
    s.run("x = 5")
    s.run("x + 1")
    assert s.render() == ">>> x = 5\n>>> x + 1\n6\n"


def test_print_output_is_captured():
    s = Session()
    s.run("print('hello')")
    assert s.render() == ">>> print('hello')\nhello\n"


def test_exception_renders_traceback():
    s = Session()
    s.run("1 / 0")
    out = s.render()
    assert ">>> 1 / 0\n" in out
    assert "Traceback (most recent call last):" in out
    assert "ZeroDivisionError: division by zero" in out


def test_multi_line_input_renders_continuation_prompts():
    s = Session()
    s.run("def f(x):\n    return x * 2")
    s.run("f(3)")
    out = s.render()
    assert ">>> def f(x):\n... " in out
    assert "    return x * 2" in out
    assert ">>> f(3)\n6\n" in out


def test_import_persists_across_runs():
    s = Session()
    s.run("import math")
    s.run("math.sqrt(16)")
    assert ">>> math.sqrt(16)\n4.0\n" in s.render()
```

- [ ] **Step 2: Run tests to confirm RED**

```bash
uv run pytest tests/test_listings_infra/test_repl_console.py -v
```
Expected: `ModuleNotFoundError: No module named 'repl_console'`.

- [ ] **Step 3: Implement `repl_console.py`**

Create `scripts/repl_console.py`:

```python
"""Render Python REPL transcripts deterministically.

Drivers create a Session and feed source strings via `run()`. The Session
executes them through a stdlib `code.InteractiveConsole`, capturing input,
output, and tracebacks in the canonical interactive-Python format
(>>> prompts, ... continuations, standard tracebacks).
"""
from __future__ import annotations

import code
import contextlib
import io


class Session:
    """A scriptable Python REPL session that renders as a >>>-prefixed transcript."""

    def __init__(self) -> None:
        self._console = code.InteractiveConsole(locals={})
        self._lines: list[str] = []

    def run(self, source: str) -> None:
        """Execute `source` (one logical block) and append its rendering."""
        self._lines.extend(self._format_input(source))

        out_buf = io.StringIO()
        # console.write is what InteractiveConsole uses for tracebacks
        orig_write = self._console.write
        self._console.write = out_buf.write
        try:
            with contextlib.redirect_stdout(out_buf):
                src_lines = source.split("\n")
                more = False
                for line in src_lines:
                    more = self._console.push(line)
                # Force completion if the console is still waiting
                if more:
                    self._console.push("")
        finally:
            self._console.write = orig_write

        output = out_buf.getvalue()
        if output:
            # Strip a single trailing newline; we'll add one when joining
            self._lines.append(output.rstrip("\n"))

    def render(self) -> str:
        """Return the full transcript as a single string ending with newline."""
        return "\n".join(self._lines) + "\n"

    @staticmethod
    def _format_input(source: str) -> list[str]:
        src_lines = source.split("\n")
        return [">>> " + src_lines[0]] + ["... " + line for line in src_lines[1:]]
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
uv run pytest tests/test_listings_infra/test_repl_console.py -v
```
Expected: 7 passed. If any traceback test fails, inspect the actual output and adjust the assertion to match what InteractiveConsole emits — the format may differ slightly across Python versions but is stable within a version.

- [ ] **Step 5: Commit**

```bash
git add scripts/repl_console.py tests/test_listings_infra/test_repl_console.py
git commit -m "feat(listings): add repl_console.Session for deterministic Python REPL transcripts"
```

---

### Task 7: Implement `build_listings.py` (build mode)

**Files:**
- Create: `tests/test_listings_infra/test_build_listings.py`
- Create: `scripts/build_listings.py`

- [ ] **Step 1: Write failing tests for the `build` mode**

Create `tests/test_listings_infra/test_build_listings.py`:

```python
"""End-to-end tests for the build_listings CLI."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_listings.py"


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUILD_SCRIPT)] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


@pytest.fixture
def tmp_repo(tmp_path):
    """A temp dir containing scripts/, a manifest, and an outputs dir."""
    (tmp_path / "scripts").mkdir()
    # Copy the real script modules into the temp repo so build_listings.py
    # can import them.
    for name in ("build_listings.py", "manifest.py", "normalize_output.py", "repl_console.py"):
        (tmp_path / "scripts" / name).write_text(
            (REPO_ROOT / "scripts" / name).read_text()
        )
    (tmp_path / "latex" / "manifests").mkdir(parents=True)
    (tmp_path / "latex" / "generated" / "testing").mkdir(parents=True)
    return tmp_path


def test_build_writes_normalized_output(tmp_repo):
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: echo\n"
        "    description: echo demo\n"
        "    command: [\"python\", \"-c\", \"print('hello in 0.5s')\"]\n"
        "    output_file: \"latex/generated/testing/echo.txt\"\n"
        "    expected_exit_code: 0\n"
        "    normalize: [pytest_timings]\n"
    )
    result = _run_cli(["build", "testing"], cwd=tmp_repo)
    assert result.returncode == 0, result.stderr
    out_file = tmp_repo / "latex" / "generated" / "testing" / "echo.txt"
    assert out_file.exists()
    assert out_file.read_text() == "hello in X.XXs\n"


def test_build_fails_when_exit_code_mismatches(tmp_repo):
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: must_pass\n"
        "    description: but actually fails\n"
        "    command: [\"python\", \"-c\", \"import sys; sys.exit(2)\"]\n"
        "    output_file: \"latex/generated/testing/must_pass.txt\"\n"
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    result = _run_cli(["build", "testing"], cwd=tmp_repo)
    assert result.returncode != 0
    assert "expected exit code 0" in (result.stderr + result.stdout).lower()


def test_build_accepts_nonzero_expected_exit(tmp_repo):
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: expect_fail\n"
        "    description: failure on purpose\n"
        "    command: [\"python\", \"-c\", \"import sys; print('boom'); sys.exit(1)\"]\n"
        "    output_file: \"latex/generated/testing/expect_fail.txt\"\n"
        "    expected_exit_code: 1\n"
        "    normalize: []\n"
    )
    result = _run_cli(["build", "testing"], cwd=tmp_repo)
    assert result.returncode == 0, result.stderr
    out_file = tmp_repo / "latex" / "generated" / "testing" / "expect_fail.txt"
    assert out_file.read_text() == "boom\n"


def test_build_only_filters_to_one_entry(tmp_repo):
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: a\n"
        "    description: a\n"
        "    command: [\"python\", \"-c\", \"print('A')\"]\n"
        "    output_file: \"latex/generated/testing/a.txt\"\n"
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
        "  - id: b\n"
        "    description: b\n"
        "    command: [\"python\", \"-c\", \"print('B')\"]\n"
        "    output_file: \"latex/generated/testing/b.txt\"\n"
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    result = _run_cli(["build", "testing", "--only", "b"], cwd=tmp_repo)
    assert result.returncode == 0, result.stderr
    assert not (tmp_repo / "latex" / "generated" / "testing" / "a.txt").exists()
    assert (tmp_repo / "latex" / "generated" / "testing" / "b.txt").read_text() == "B\n"
```

- [ ] **Step 2: Run tests to confirm RED**

```bash
uv run pytest tests/test_listings_infra/test_build_listings.py -v
```
Expected: all fail (`build_listings.py` doesn't exist yet, or exists but doesn't do what the tests assert).

- [ ] **Step 3: Implement `build_listings.py` (build mode + --only)**

Create `scripts/build_listings.py`:

```python
#!/usr/bin/env python3
"""CLI to build/verify generated listing outputs from a chapter manifest.

Usage:
    python scripts/build_listings.py build <chapter> [--only ID]
    python scripts/build_listings.py verify <chapter> [--only ID]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Allow imports of sibling modules whether invoked as a script or via -m
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import Manifest, ManifestEntry, load_manifest  # noqa: E402
from normalize_output import apply_normalizers  # noqa: E402


def _manifest_path(chapter: str) -> Path:
    return Path("latex") / "manifests" / f"{chapter}.yaml"


def _run_entry(entry: ManifestEntry) -> str:
    """Execute one manifest entry and return its normalized output."""
    if entry.driver is not None:
        cmd = [sys.executable, entry.driver]
    else:
        cmd = list(entry.command)  # type: ignore[arg-type]

    proc = subprocess.run(
        cmd,
        cwd=entry.cwd,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    if proc.returncode != entry.expected_exit_code:
        raise SystemExit(
            f"[{entry.id}] expected exit code {entry.expected_exit_code}, "
            f"got {proc.returncode}\n--- captured output ---\n{combined}"
        )
    return apply_normalizers(combined, entry.normalize)


def cmd_build(manifest: Manifest, only: str | None) -> int:
    for entry in manifest.outputs:
        if only is not None and entry.id != only:
            continue
        text = _run_entry(entry)
        out_path = Path(entry.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"wrote {entry.output_file}")
    return 0


def cmd_verify(manifest: Manifest, only: str | None) -> int:
    # Implemented in Task 8.
    raise NotImplementedError("verify is implemented in Task 8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_listings")
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("build", "verify"):
        sp = sub.add_parser(mode)
        sp.add_argument("chapter")
        sp.add_argument("--only", default=None)

    args = parser.parse_args(argv)
    manifest = load_manifest(_manifest_path(args.chapter))
    if args.mode == "build":
        return cmd_build(manifest, args.only)
    if args.mode == "verify":
        return cmd_verify(manifest, args.only)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
uv run pytest tests/test_listings_infra/test_build_listings.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_listings.py tests/test_listings_infra/test_build_listings.py
git commit -m "feat(listings): build_listings.py with build mode and --only filter"
```

---

### Task 8: Add `verify` mode to `build_listings.py`

**Files:**
- Modify: `tests/test_listings_infra/test_build_listings.py`
- Modify: `scripts/build_listings.py`

- [ ] **Step 1: Write failing tests for verify**

Append to `tests/test_listings_infra/test_build_listings.py`:

```python
def test_verify_passes_when_committed_matches(tmp_repo):
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: e\n"
        "    description: e\n"
        "    command: [\"python\", \"-c\", \"print('E')\"]\n"
        "    output_file: \"latex/generated/testing/e.txt\"\n"
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    # Pre-commit the expected output
    (tmp_repo / "latex" / "generated" / "testing" / "e.txt").write_text("E\n")
    result = _run_cli(["verify", "testing"], cwd=tmp_repo)
    assert result.returncode == 0, result.stderr


def test_verify_fails_with_diff_when_drift(tmp_repo):
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: e\n"
        "    description: e\n"
        "    command: [\"python\", \"-c\", \"print('NEW')\"]\n"
        "    output_file: \"latex/generated/testing/e.txt\"\n"
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    # Committed output is stale
    (tmp_repo / "latex" / "generated" / "testing" / "e.txt").write_text("OLD\n")
    result = _run_cli(["verify", "testing"], cwd=tmp_repo)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "DRIFT" in output
    assert "-OLD" in output or "OLD" in output
    assert "+NEW" in output or "NEW" in output
    assert "build testing --only e" in output


def test_verify_fails_when_output_file_missing(tmp_repo):
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: missing\n"
        "    description: missing committed\n"
        "    command: [\"python\", \"-c\", \"print('X')\"]\n"
        "    output_file: \"latex/generated/testing/missing.txt\"\n"
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    result = _run_cli(["verify", "testing"], cwd=tmp_repo)
    assert result.returncode != 0
    assert "missing" in (result.stdout + result.stderr).lower()
```

- [ ] **Step 2: Run tests to confirm RED**

```bash
uv run pytest tests/test_listings_infra/test_build_listings.py -v
```
Expected: the three new tests fail (`NotImplementedError` or similar).

- [ ] **Step 3: Implement `cmd_verify`**

In `scripts/build_listings.py`, replace the `cmd_verify` stub with:

```python
import difflib  # add at top with other imports


def cmd_verify(manifest: Manifest, only: str | None) -> int:
    drift_count = 0
    for entry in manifest.outputs:
        if only is not None and entry.id != only:
            continue
        regenerated = _run_entry(entry)
        out_path = Path(entry.output_file)
        if not out_path.exists():
            print(
                f"DRIFT in {entry.output_file}: file missing\n"
                f"Fix: python scripts/build_listings.py build "
                f"{manifest.chapter} --only {entry.id}",
                file=sys.stderr,
            )
            drift_count += 1
            continue
        committed = out_path.read_text()
        if committed != regenerated:
            diff = "".join(
                difflib.unified_diff(
                    committed.splitlines(keepends=True),
                    regenerated.splitlines(keepends=True),
                    fromfile="committed",
                    tofile="regenerated",
                )
            )
            print(
                f"DRIFT in {entry.output_file}\n{diff}\n"
                f"Fix: python scripts/build_listings.py build "
                f"{manifest.chapter} --only {entry.id}",
                file=sys.stderr,
            )
            drift_count += 1
    if drift_count:
        return 1
    print(f"verify: {len(manifest.outputs)} outputs OK")
    return 0
```

- [ ] **Step 4: Run tests to verify GREEN**

```bash
uv run pytest tests/test_listings_infra/test_build_listings.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_listings.py tests/test_listings_infra/test_build_listings.py
git commit -m "feat(listings): build_listings.py verify mode with unified diff"
```

---

## Phase B — Catalog and pre-checks

### Task 9: Catalog every `lstlisting` block in `book-testing.tex`

**Files:**
- Create: `latex/CONVERSION_NOTES.md`

- [ ] **Step 1: Enumerate every block**

Run:
```bash
grep -n "begin{lstlisting}\|end{lstlisting}" latex/book-testing.tex > /tmp/blocks.txt
wc -l /tmp/blocks.txt
```
Expected: 316 lines (158 begin + 158 end pairs).

- [ ] **Step 2: Build the catalog**

Read `latex/book-testing.tex` block-by-block and create `latex/CONVERSION_NOTES.md` with a table classifying each block. Structure:

```markdown
# Testing chapter conversion notes

Running log of conversion decisions, edge cases, and patterns discovered
during conversion of `book-testing.tex`. Becomes raw material for the
future `converting-chapter-listings` skill.

## Catalog

| # | Lines | Style | Type | Planned action | Notes |
|---|-------|-------|------|----------------|-------|
| 1 | 71–88 | Python | code | extract → `src/bettercode/testing/escape_velocity_v1.py` (full file) | initial version, no validation; book shows `import numpy as np` which v1 file does not need — defer decision (Pattern strategy choice) |
| 2 | 92–102 | Python | code | extract → `tests/testing_chapter/test_escape_velocity.py::test_escape_velocity` (lines 6–14) | imports `escape_velocity` from v1; tests the no-validation version |
| 3 | 108–110 | Pythonshort | inline-keep | keep inline; syntax pattern only | |
| 4 | 116–123 | shell | output | manifest entry: `pytest_escape_velocity_pass` | pytest output for block 2 test |
| ... | ... | ... | ... | ... | ... |
```

For each block, record:
- **Lines**: the line range in `book-testing.tex` between `\begin{lstlisting}` and `\end{lstlisting}`.
- **Style**: the style argument (`Python`, `Pythonshort`, `shell`, `repl`, `R`, ...).
- **Type**: `code` (runnable Python), `output` (program output to capture), `inline-keep` (syntax pattern / tiny block).
- **Planned action**:
  - `code` → name + path of the snippet/test file it will live in. Note if multiple blocks slice from the same file.
  - `output` → manifest entry id and whether it's a pytest command or a REPL driver.
  - `inline-keep` → just "keep inline".
- **Notes**: any inter-block dependencies, import mismatches, intentionally-broken state.

- [ ] **Step 3: Add a "patterns observed" section**

After the catalog table, add:

```markdown
## Patterns observed during cataloging

(Fill in as each block is cataloged. Examples:)
- **Inter-block pattern 1 (function + test):** N occurrences. Resolution: separate `src/bettercode/testing/*.py` and `tests/testing_chapter/test_*.py` files.
- **Pedagogical import in function block (`import numpy as np` shown but unused):** N occurrences. Resolution chosen: ...
- **Multi-block sequence from one file (line ranges):** N occurrences.
- **REPL session split into multiple visual blocks:** N occurrences.

## Open questions raised during cataloging

- (record any blocks where the right approach is unclear — discuss with author before resolving)
```

- [ ] **Step 4: Commit the catalog**

```bash
git add latex/CONVERSION_NOTES.md
git commit -m "docs(testing): catalog all 158 lstlisting blocks for conversion"
```

- [ ] **Step 5: Pause for author review**

Tell the author the catalog is ready and the rest of the migration depends on the resolutions of any items in the "Open questions" section. Wait for confirmation before continuing.

---

### Task 10: Re-run the external-importer pre-check

**Files:**
- Modify: `latex/CONVERSION_NOTES.md`

- [ ] **Step 1: Grep for external imports of every module being touched**

For each `src/bettercode/<module>` referenced in the catalog (likely: `escape_velocity`, `bug_driven_testing`, `distance`, `simpleScaler`, `my_linear_regression`, `narps.bids_utils`, `textmining`, plus any others), run:

```bash
for mod in escape_velocity bug_driven_testing distance simpleScaler my_linear_regression narps textmining; do
  echo "=== $mod ==="
  grep -rn "from bettercode.$mod\|import bettercode.$mod" \
    /Users/poldrack/Dropbox/code/BetterCodeBetterScience/bettercode/notebooks/ \
    /Users/poldrack/Dropbox/code/BetterCodeBetterScience/example-* 2>/dev/null
done
```

- [ ] **Step 2: Record results in CONVERSION_NOTES.md**

Append a section to `latex/CONVERSION_NOTES.md`:

```markdown
## External-importer pre-check (re-run YYYY-MM-DD)

Result of grep over `notebooks/` and `../example-*/` for each module being migrated:

| Module | External importers | Safe to rename? |
|--------|---------------------|-----------------|
| escape_velocity | (none) | yes |
| bug_driven_testing | (none) | yes |
| ... | ... | ... |
```

If any module has external importers, STOP and consult the author before proceeding. The plan as written assumes none; the spec's "Caveat for future chapters" addresses the alternative strategies.

- [ ] **Step 3: Commit**

```bash
git add latex/CONVERSION_NOTES.md
git commit -m "docs(testing): record external-importer pre-check results"
```

---

## Phase C — Migrate code and tests

### Task 11: Create the `src/bettercode/testing/` package skeleton

**Files:**
- Create: `src/bettercode/testing/__init__.py` (empty)

- [ ] **Step 1: Create empty `__init__.py`**

```bash
mkdir -p src/bettercode/testing
: > src/bettercode/testing/__init__.py
```

- [ ] **Step 2: Verify package imports**

```bash
uv run python -c "import bettercode.testing; print(bettercode.testing.__file__)"
```
Expected: prints the path to the new `__init__.py`.

- [ ] **Step 3: Commit**

```bash
git add src/bettercode/testing/__init__.py
git commit -m "feat(testing): scaffold bettercode.testing package"
```

---

### Task 12: Create snippet files in batches; one batch per commit

**Files (per batch):**
- Create / move: files under `src/bettercode/testing/` as enumerated in `CONVERSION_NOTES.md`

This task is iterative — process snippet files in batches of ~10 (or one conceptual group per batch, whichever is smaller). Each batch produces one commit.

For each snippet file you create, follow this loop:

- [ ] **Step 1: Pick the next un-migrated batch** from `CONVERSION_NOTES.md`'s catalog.

- [ ] **Step 2: For each entry in the batch**:
  - If it's a rename of an existing module (e.g., `src/bettercode/escape_velocity.py` → `src/bettercode/testing/escape_velocity_v2.py`):
    ```bash
    git mv src/bettercode/escape_velocity.py src/bettercode/testing/escape_velocity_v2.py
    ```
  - If it's a new pedagogical version (e.g., `escape_velocity_v1.py`):
    Create the file by hand to match the exact text shown in the LaTeX block, plus whatever imports are needed for it to be runnable on its own.
  - If it's a "broken on purpose" version (e.g., `find_outliers_v1_buggy.py`):
    Create the file containing the buggy code as shown.

- [ ] **Step 3: Verify each new/moved file at least imports cleanly**

```bash
uv run python -c "import importlib; importlib.import_module('bettercode.testing.<modname>')"
```
Repeat for each module in the batch. Expected: no exception (a broken-on-purpose module must still parse and import — exceptions are only raised when its function is called with the bad inputs).

- [ ] **Step 4: Mark the batch as done in `CONVERSION_NOTES.md`**

In the catalog table, prepend ✅ to the row's `#` for each completed entry.

- [ ] **Step 5: Commit the batch**

```bash
git add src/bettercode/testing/ latex/CONVERSION_NOTES.md
git commit -m "feat(testing): add canonical snippet files for blocks N–M"
```

- [ ] **Step 6: Repeat** from Step 1 until every `code`-typed row in the catalog has a ✅.

---

### Task 13: Migrate test files to `tests/testing_chapter/`

**Files:**
- Create: `tests/testing_chapter/__init__.py` (empty)
- Move / create: `tests/testing_chapter/test_*.py` as enumerated in the catalog
- Modify: any docs that pointed at the old test paths

- [ ] **Step 1: Create the package skeleton**

```bash
mkdir -p tests/testing_chapter
: > tests/testing_chapter/__init__.py
```

- [ ] **Step 2: Move existing tests featured in the chapter**

```bash
git mv tests/test_escape_velocity.py tests/testing_chapter/test_escape_velocity.py
git mv tests/test_find_outliers.py tests/testing_chapter/test_find_outliers.py
```

(Add any other tests the catalog identifies as belonging to this chapter.)

- [ ] **Step 3: Update imports in each moved test**

Edit each moved file to import from the new `bettercode.testing.*` modules. Example for `tests/testing_chapter/test_escape_velocity.py`:

Change:
```python
from bettercode.escape_velocity import escape_velocity
```
to (assuming v2 is the version under test for the passing case):
```python
from bettercode.testing.escape_velocity_v2 import escape_velocity
```

For tests that exercise the v1 (no-validation) version specifically, import from `bettercode.testing.escape_velocity_v1` instead.

- [ ] **Step 4: Create the failing-on-purpose demo test files**

For each demo block in the catalog where the chapter shows a *failing* pytest run (e.g., the `ev_expected = 1186.0` typo), create a dedicated test file. Example: `tests/testing_chapter/test_escape_velocity_typo.py`:

```python
"""Demonstration test that fails on purpose (book shows the failure output)."""
import numpy as np
from bettercode.testing.escape_velocity_v1 import escape_velocity


def test_escape_velocity_with_typo():
    mass_earth = 5.972e24
    radius_earth = 6.371e6
    ev_expected = 1186.0  # intentional typo — real value is 11186.0
    ev_computed = escape_velocity(mass_earth, radius_earth)
    assert np.allclose(ev_expected, ev_computed), "Test failed!"
```

Mark these files (or the demo tests within them) appropriately so they don't break ordinary `pytest` runs. Two acceptable approaches — pick one and apply consistently across all demo failure files:
- Move the file under `tests/testing_chapter/_demos/` and configure pytest to skip that directory in normal runs (via `[tool.pytest.ini_options] norecursedirs` or a `conftest.py` `collect_ignore`).
- Decorate every demo test with `@pytest.mark.xfail(strict=True)` so they "pass" because they fail as expected.

Record the chosen convention in `CONVERSION_NOTES.md`.

- [ ] **Step 5: Run the full test suite to confirm nothing is broken outside the demos**

```bash
uv run pytest -q
```
Expected: every previously-passing test still passes. Demo failure files are either uncollected (option 1) or pass via xfail (option 2).

- [ ] **Step 6: Commit**

```bash
git add tests/testing_chapter/ tests/test_escape_velocity.py tests/test_find_outliers.py latex/CONVERSION_NOTES.md pyproject.toml
git commit -m "feat(testing): migrate chapter tests to tests/testing_chapter/ and add demo-failure tests"
```

---

## Phase D — REPL drivers and manifest

### Task 14: Write REPL driver scripts

**Files:**
- Create: `scripts/repl_drivers/__init__.py` (empty, optional — see note)
- Create: `scripts/repl_drivers/testing/<id>.py` for each REPL-style row in the catalog

> Note on package status: `scripts/` is not a Python package and `scripts/repl_drivers/` doesn't need to be either — these are standalone scripts invoked by path. Skip the `__init__.py` files unless something else demands them.

- [ ] **Step 1: Create the directory**

```bash
mkdir -p scripts/repl_drivers/testing
```

- [ ] **Step 2: For each REPL block in the catalog, create one driver**

Each driver follows this skeleton (example: `scripts/repl_drivers/testing/floating_point_allclose.py`):

```python
#!/usr/bin/env python3
"""REPL driver: floating-point comparison with np.allclose.

Renders the IPython-replacement transcript shown in book-testing.tex around
the discussion of `np.allclose` for the np.allclose example.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repl_console import Session


def main() -> None:
    s = Session()
    s.run("import numpy as np")
    s.run("np.allclose(0.1 + 0.2, 0.3)")
    sys.stdout.write(s.render())


if __name__ == "__main__":
    main()
```

Per the spec's Pattern 3a, each visual REPL block in the chapter gets its own driver. Re-import / re-define what each block needs so drivers are independently regenerable.

- [ ] **Step 3: Spot-check one driver's output**

```bash
uv run python scripts/repl_drivers/testing/floating_point_allclose.py
```
Expected: prints
```
>>> import numpy as np
>>> np.allclose(0.1 + 0.2, 0.3)
True
```

- [ ] **Step 4: Commit when a batch of drivers is complete**

```bash
git add scripts/repl_drivers/
git commit -m "feat(testing): add REPL drivers for chapter blocks N–M"
```

Repeat for all REPL blocks. Record any driver-authoring gotchas in `CONVERSION_NOTES.md`.

---

### Task 15: Write `latex/manifests/testing.yaml`

**Files:**
- Create: `latex/manifests/testing.yaml`

- [ ] **Step 1: Create the manifest skeleton**

Create `latex/manifests/testing.yaml`:

```yaml
chapter: testing
outputs: []
```

- [ ] **Step 2: Add one entry per `output`-typed row in the catalog**

Iterate through `CONVERSION_NOTES.md`. For each `output` row, add a manifest entry. Two shapes:

**Pytest entry:**
```yaml
  - id: pytest_escape_velocity_pass
    description: "Passing pytest output for test_escape_velocity"
    command: ["uv", "run", "pytest",
              "tests/testing_chapter/test_escape_velocity.py::test_escape_velocity"]
    output_file: "latex/generated/testing/pytest_escape_velocity_pass.txt"
    expected_exit_code: 0
    normalize: [pytest_timings, file_paths, terminal_width]
```

**REPL driver entry:**
```yaml
  - id: repl_floating_point_allclose
    description: "REPL: np.allclose for floating-point comparison"
    driver: "scripts/repl_drivers/testing/floating_point_allclose.py"
    output_file: "latex/generated/testing/repl_floating_point_allclose.txt"
    expected_exit_code: 0
    normalize: []
```

For pytest entries that show failures, set `expected_exit_code: 1` and add `memory_addresses` to `normalize` (pytest failure summaries often include `<function ... at 0x...>` lines).

- [ ] **Step 3: Validate the manifest parses**

```bash
uv run python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from manifest import load_manifest
m = load_manifest(Path('latex/manifests/testing.yaml'))
print(f'{m.chapter}: {len(m.outputs)} entries')
for e in m.outputs:
    print(f'  {e.id} -> {e.output_file}')
"
```
Expected: prints the chapter name and one line per entry with no exceptions.

- [ ] **Step 4: Commit**

```bash
git add latex/manifests/testing.yaml
git commit -m "feat(testing): add listings manifest for the testing chapter"
```

---

## Phase E — Generate outputs and rewrite LaTeX

### Task 16: Generate all outputs

**Files:**
- Create: `latex/generated/testing/*.txt`

- [ ] **Step 1: Run the full chapter build**

```bash
uv run python scripts/build_listings.py build testing
```
Expected: prints one `wrote latex/generated/testing/<id>.txt` line per manifest entry. Exits 0.

If any entry exits with the wrong code, fix the underlying snippet/test/driver first.

- [ ] **Step 2: Spot-check a handful of generated files**

```bash
ls latex/generated/testing/
head -30 latex/generated/testing/pytest_escape_velocity_pass.txt
head -30 latex/generated/testing/repl_floating_point_allclose.txt
```
Expected: every manifest output_file present; contents look like the corresponding block in the original chapter (minus timing/address noise).

- [ ] **Step 3: Commit**

```bash
git add latex/generated/testing/
git commit -m "feat(testing): generate captured outputs for the chapter manifest"
```

---

### Task 17: Rewrite `book-testing.tex` in batches

**Files:**
- Modify: `latex/book-testing.tex`

This task is iterative — rewrite in chunks of ~20 blocks per commit so any rendering regression can be bisected.

- [ ] **Step 1: For each block in the catalog**:
  - If type is `code`: replace the entire `\begin{lstlisting}[style=...]...\end{lstlisting}` block with:
    ```latex
    \lstinputlisting[style=Python]{../src/bettercode/testing/<file>.py}
    ```
    Add `firstline=N, lastline=M` if the catalog says only part of the file should be shown.
  - If type is `output`: replace with:
    ```latex
    \lstinputlisting[style=shell]{generated/testing/<id>.txt}
    ```
    (use the manifest entry's `output_file` path, made relative to `latex/`).
  - If type is `inline-keep`: leave unchanged.

- [ ] **Step 2: After each chunk, rebuild the PDF**

```bash
cd latex && make pdf-no-verify 2>&1 | tail -30
```
(`pdf-no-verify` is added in Task 18; use `make all` until then.) Expected: PDF builds; visually scan affected pages for obvious regressions.

- [ ] **Step 3: Commit the chunk**

```bash
git add latex/book-testing.tex
git commit -m "refactor(testing): convert blocks N–M to \\lstinputlisting references"
```

- [ ] **Step 4: Repeat** until every catalog row is converted.

- [ ] **Step 5: Final verify pass**

```bash
uv run python scripts/build_listings.py verify testing
```
Expected: `verify: N outputs OK`.

```bash
cd latex && make
```
Expected: PDF builds without LaTeX errors.

---

### Task 18: Visual diff and renderer tweaks

**Files:**
- Modify (as needed): `latex/book-testing.tex`, `latex/designchanges.sty`, individual `src/bettercode/testing/*.py` files

- [ ] **Step 1: Diff the rendered PDF against the prior version**

```bash
git log --oneline latex/book.pdf | head -5    # find a prior commit
git show <prior-sha>:latex/book.pdf > /tmp/old_book.pdf
# Compare /tmp/old_book.pdf and latex/book.pdf visually (open both, scan for differences)
```

Look specifically for:
- Long lines in code blocks that overflow the right margin (fix with `breaklines=true` on the listing or the style).
- REPL transcripts: confirm `>>>` prompt formatting is acceptable. If not, edit `\lstdefinestyle{repl}` in `latex/designchanges.sty` to set `frame`, `numbers`, or padding appropriately for the new prompt convention.
- Pytest failure blocks: confirm normalized line ranges read sensibly.

- [ ] **Step 2: Iterate on `lstinputlisting` options and normalizers**

For any block that renders poorly, either:
- Add `firstline`/`lastline`/`breaklines` options to the `\lstinputlisting` call, or
- Add or refine a normalizer (with a new test under `tests/test_listings_infra/test_normalize_output.py`), then re-run `build_listings.py build testing` and re-verify.

Record the trickier fixes in `CONVERSION_NOTES.md`.

- [ ] **Step 3: Add chapter intro sentence about the REPL convention**

In `latex/book-testing.tex`, near the first occurrence of `>>>` (or in the chapter intro), add one sentence such as:

> Throughout this book, interactive Python sessions are shown using the standard Python REPL prompts (\texttt{>>>} for input, \texttt{...} for continuation).

- [ ] **Step 4: Commit any tweaks**

```bash
git add latex/book-testing.tex latex/designchanges.sty src/bettercode/testing/ latex/generated/testing/ scripts/normalize_output.py tests/test_listings_infra/
git commit -m "polish(testing): renderer tweaks and normalizer refinements"
```

---

## Phase F — Build automation

### Task 19: Update `latex/Makefile`

**Files:**
- Modify: `latex/Makefile`

- [ ] **Step 1: Add `listings`, `verify-listings`, and `pdf-no-verify` targets**

Edit `latex/Makefile` to add these targets and gate the existing PDF target on `verify-listings`:

```makefile
MAIN = book
LATEXMK = latexmk
LATEXMK_FLAGS = -xelatex -f # -pdf -interaction=nonstopmode -halt-on-error

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex verify-listings
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN)

# Regenerate captured listing outputs from the testing chapter manifest
listings:
	cd .. && uv run python scripts/build_listings.py build testing

# Verify committed listing outputs still match what the code would produce.
# This is what gates the PDF build.
verify-listings:
	cd .. && uv run python scripts/build_listings.py verify testing

# Escape hatch for prose-only iteration: build the PDF without verifying listings.
pdf-no-verify: $(MAIN).tex
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN)

# Continuous build — rebuilds on file changes
watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(MAIN)

clean:
	$(LATEXMK) -c

distclean:
	$(LATEXMK) -C

.PHONY: all watch clean distclean listings verify-listings pdf-no-verify
```

- [ ] **Step 2: Smoke test the new targets**

```bash
cd latex && make verify-listings
```
Expected: `verify: N outputs OK`.

```bash
cd latex && make listings
```
Expected: regenerates outputs (no diff against current commit if no drift).

```bash
cd latex && make
```
Expected: runs verify-listings, then builds the PDF.

- [ ] **Step 3: Commit**

```bash
git add latex/Makefile
git commit -m "build(latex): gate pdf build on verify-listings; add listings target"
```

---

### Task 20: Add CI workflow

**Files:**
- Create: `.github/workflows/verify-listings.yml`

- [ ] **Step 1: Check the existing workflows directory**

```bash
ls .github/workflows/ 2>/dev/null
```
Note what's there so the new file matches house style.

- [ ] **Step 2: Create the workflow file**

Create `.github/workflows/verify-listings.yml`:

```yaml
name: Verify chapter listings

on:
  pull_request:
    paths:
      - "latex/**"
      - "src/bettercode/testing/**"
      - "tests/testing_chapter/**"
      - "scripts/**"
      - "latex/manifests/**"
      - "pyproject.toml"
      - "uv.lock"
  push:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.12

      - name: Sync dependencies
        run: uv sync

      - name: Verify testing-chapter listings
        run: uv run python scripts/build_listings.py verify testing
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/verify-listings.yml
git commit -m "ci: verify testing-chapter listings on push/PR"
```

- [ ] **Step 4: Push and watch the first CI run**

After pushing, watch the workflow run on GitHub. If it fails due to environment differences (e.g., the `terminal_width` normalizer didn't fully stabilize line lengths), either tighten the normalizer or regenerate the affected outputs in CI's environment and commit them. Record any CI-specific normalization needed in `CONVERSION_NOTES.md`.

---

### Task 21: Final spec-coverage sweep

**Files:**
- Modify (if needed): `docs/superpowers/specs/2026-05-27-latex-reproducible-listings-design.md`
- Modify: `latex/CONVERSION_NOTES.md`

- [ ] **Step 1: Re-read the spec section by section**

For each spec section (Problem, Goals, Architectural decisions, Directory layout, Manifest schema, Build script, Normalizers, REPL transcript generation, Cross-block context, Workflow, CI, Migration plan, Risks, Extensibility, Follow-up work), confirm the corresponding implementation exists.

- [ ] **Step 2: Confirm all five Goals from the spec are met**

```
1. Every non-trivial code block resolves to a runnable canonical file.
2. Every shell/REPL output block is generated by actually running a command.
3. Build hard-fails if outputs drift.
4. --only ID supports single-block iteration.
5. Design extends cleanly to subsequent chapters with no tooling changes.
```

For each goal, identify a specific artifact that proves it (e.g., for #3, point at `make pdf` failing when `latex/generated/testing/pytest_escape_velocity_pass.txt` is manually edited).

- [ ] **Step 3: Finalize `CONVERSION_NOTES.md`**

Add a closing section:

```markdown
## Conversion complete

Date: YYYY-MM-DD
Blocks converted: N of 158
Inline-kept blocks: M

## Patterns to capture in `converting-chapter-listings` skill

(Distilled list of decision rules, gotchas, naming conventions, and
recipes discovered during this conversion — bullet form, suitable for
direct lift into the future skill.)
```

- [ ] **Step 4: Commit**

```bash
git add latex/CONVERSION_NOTES.md
git commit -m "docs(testing): finalize conversion notes for testing chapter"
```

- [ ] **Step 5: Hand off to the author**

Report:
- Number of blocks converted vs. kept inline.
- Number of manifest entries.
- A list of any open questions or follow-up items recorded during the work.
- Confirmation that `make` builds a PDF and `make verify-listings` exits clean.

---

## Self-review (filled in)

**Spec coverage:**
- Problem statement → Tasks 1–20 address the underlying drift problem.
- Goals 1–5 → Goal #1 (Tasks 11–13), #2 (Tasks 6, 14, 16), #3 (Task 19), #4 (Tasks 7–8), #5 (architecture preserved throughout).
- Architectural decisions 1–9 → all implemented in Tasks 1–19.
- Directory layout → Tasks 2, 11, 13, 14, 15, 16.
- Manifest schema → Task 5 (loader), Task 15 (chapter manifest).
- Build script → Tasks 7, 8.
- Normalizers → Tasks 3, 4.
- REPL transcript generation → Task 6 (helper), Task 14 (drivers).
- Cross-block context patterns → catalog (Task 9) records per-block resolution; snippet/test creation (Tasks 12–13) realizes them.
- Workflow / Makefile → Task 19.
- CI → Task 20.
- Migration plan steps 1–11 → mapped: step 1 (Task 9), step 2 (Task 10), step 3 (Tasks 11–12), step 4 (Task 13), step 5 (Task 14), step 6 (Task 15), step 7 (Phase A), step 8 (Task 16), step 9 (Task 17), step 10 (Task 18), step 11 (running throughout).
- Risks → addressed via normalizers, CI, the catalog process, and the verify gate.
- Follow-up: skill authoring is explicitly out of scope; `CONVERSION_NOTES.md` accumulates raw material.

**Placeholders:** Spot-checked — no TBDs, no "implement later", no "similar to Task N" references. Every code step has the actual code; every command has the actual command.

**Type consistency:**
- `Session` (class) used consistently in Task 6 and Task 14 driver template.
- `Manifest` / `ManifestEntry` / `load_manifest` used consistently in Tasks 5, 7, 8, 15.
- `apply_normalizers(text, names)` signature consistent in Tasks 3, 4, 7.
- `NORMALIZERS` dict referenced consistently in Tasks 3, 4.
- CLI commands `build testing` / `verify testing` / `--only ID` consistent in Tasks 7, 8, 17, 19, 20.
