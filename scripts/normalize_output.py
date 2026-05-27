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
