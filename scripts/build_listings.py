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
