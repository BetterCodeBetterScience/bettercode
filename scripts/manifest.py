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
                f"entry {self.id!r}: exactly one of 'command' or 'driver' must be set"
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
