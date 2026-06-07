#!/usr/bin/env python3
"""CLI to build/verify generated listing outputs from a chapter manifest.

Usage:
    python scripts/build_listings.py build <chapter> [--only ID]
    python scripts/build_listings.py verify <chapter> [--only ID]
"""

from __future__ import annotations

import argparse
import difflib
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Allow imports of sibling modules whether invoked as a script or via -m
sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import Manifest, ManifestEntry, load_manifest  # noqa: E402
from normalize_output import apply_normalizers  # noqa: E402


def _manifest_path(chapter: str) -> Path:
    return Path("latex") / "manifests" / f"{chapter}.yaml"


def _command_prompt(cmd: list[str]) -> str | None:
    """Return a `$ pytest ...` shell-prompt line for a pytest command.

    The displayed command starts at the `pytest` token, dropping any runner
    wrapper such as `uv run` or `python -m`. Returns None for non-pytest
    commands, whose output is shown without a prompt line.
    """
    if "pytest" not in cmd:
        return None
    tokens = cmd[cmd.index("pytest"):]
    return "$ " + shlex.join(tokens)


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
    prompt = _command_prompt(cmd)
    if prompt is not None:
        combined = f"{prompt}\n{combined}"
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


_PYTEST_TIMING_RE = re.compile(r"\b\d+\.\d+s\b")


def _verify_canonicalize(text: str) -> str:
    """Apply verify-only normalization for comparison.

    Lets the committed output preserve volatile content (like wall-clock
    pytest timings) while still flagging substantive drift. The committed
    file shows actual seconds; verify masks them for the comparison only.
    """
    return _PYTEST_TIMING_RE.sub("X.XXs", text)


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
        if _verify_canonicalize(committed) != _verify_canonicalize(regenerated):
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
