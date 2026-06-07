"""Named text normalizers applied to captured command output.

Each normalizer is a pure `str -> str` function registered in NORMALIZERS
by name. Manifests reference normalizers by these names.
"""

from __future__ import annotations

import os
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


@register("strip_testing_chapter_dir")
def strip_testing_chapter_dir(text: str) -> str:
    """Remove the 'tests/testing_chapter/' path prefix wherever it appears."""
    return text.replace("tests/testing_chapter/", "")


@register("memory_addresses")
def memory_addresses(text: str) -> str:
    """Replace hex memory addresses like '0x101403370' with '0xADDR'."""
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)


@register("file_paths")
def file_paths(text: str) -> str:
    """Rewrite absolute paths under the current working directory to be repo-relative.

    Two cases:
    - `<cwd>/<rest>` becomes `<rest>` (e.g. pytest traceback file paths)
    - `<cwd>` standing alone becomes `.` (e.g. pytest's `rootdir:` line)
    """
    cwd = os.getcwd()
    escaped = re.escape(cwd)
    # Strip cwd prefix from paths that continue with `/`
    text = re.sub(escaped + r"/", "", text)
    # Replace standalone cwd (not followed by anything path-like) with `.`
    text = re.sub(escaped + r"(?![/\w])", ".", text)
    return text


@register("temp_paths")
def temp_paths(text: str) -> str:
    """Collapse temp-directory paths to '/tmp/PATH'."""
    # macOS: /var/folders/.../T/... - keep file extension if present
    text = re.sub(r"/var/folders/[^\s\"']*?(?=\.\w+\b|$|\s)", "/tmp/PATH", text)
    # Linux/pytest: /tmp/pytest-of-*/...
    text = re.sub(r"/tmp/pytest-[^\s\"']+", "/tmp/PATH", text)
    return text


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


@register("worker_count")
def worker_count(text: str) -> str:
    """Replace xdist worker count like '16 workers' with 'N workers'."""
    return re.sub(r"\b\d+ workers\b", "N workers", text)


@register("strip_docstrings")
def strip_docstrings(text: str) -> str:
    """Remove triple-quoted docstrings from rendered Python source.

    Matches an indented line starting with `\"\"\"` through its closing
    `\"\"\"`, including the trailing newline. Handles both single-line
    and multi-line docstrings. Defense-in-depth pair with `pytest --tb=short`:
    --tb=short usually removes function source entirely, but coverage and
    other tools may still surface docstrings.
    """
    return re.sub(
        r'^(\s+)""".*?"""\s*\n',
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )


@register("xdist_schedule")
def xdist_schedule(text: str) -> str:
    """Canonicalize xdist scheduling output to make it deterministic.

    Finds the block between 'scheduling tests via LoadScheduling' and the
    trailing summary line, normalizes worker IDs ([gwN]), and sorts the
    scheduling/PASSED lines so ordering is stable across runs.
    """
    marker = "scheduling tests via LoadScheduling"
    if marker not in text:
        return text

    lines = text.split("\n")
    start_idx = None
    for i, line in enumerate(lines):
        if marker in line:
            start_idx = i + 1
            break

    if start_idx is None:
        return text

    # Find the end: first line that is a pytest separator (starts with '===')
    # after the scheduling block. This matches both pure-'=' lines and lines
    # like '====== 10 passed ... ======'.
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        if lines[i].startswith("==="):
            end_idx = i
            break

    schedule_lines = lines[start_idx:end_idx]

    # Normalize worker IDs and completion percentages, then sort
    def _norm_line(ln: str) -> str:
        ln = re.sub(r"\[gw\d+\]", "[gwN]", ln)
        ln = re.sub(r"\[ *\d+%\]", "[NNN%]", ln)
        return ln

    normalized = [_norm_line(ln) for ln in schedule_lines]
    sorted_lines = sorted(normalized)

    result_lines = lines[:start_idx] + sorted_lines + lines[end_idx:]
    return "\n".join(result_lines)


def apply_normalizers(text: str, names: list[str]) -> str:
    """Apply each named normalizer in order. Unknown names raise KeyError."""
    for name in names:
        text = NORMALIZERS[name](text)
    return text
