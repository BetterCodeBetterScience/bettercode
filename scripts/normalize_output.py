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


@register("memory_addresses")
def memory_addresses(text: str) -> str:
    """Replace hex memory addresses like '0x101403370' with '0xADDR'."""
    return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)


@register("file_paths")
def file_paths(text: str) -> str:
    """Rewrite absolute paths under the current working directory to be repo-relative."""
    cwd = os.getcwd()
    # Use a regex that matches quoted or unquoted absolute paths under cwd.
    # cwd may contain regex metachars; escape it.
    pattern = re.compile(re.escape(cwd) + r"/")
    return pattern.sub("", text)


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


def apply_normalizers(text: str, names: list[str]) -> str:
    """Apply each named normalizer in order. Unknown names raise KeyError."""
    for name in names:
        text = NORMALIZERS[name](text)
    return text
