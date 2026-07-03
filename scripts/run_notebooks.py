#!/usr/bin/env python3
"""Execute the chapter notebooks and report which succeed and which fail.

Runs every ``ch-*/*.ipynb`` under a root directory (default ``notebooks``) with
``jupyter nbconvert --execute`` into a throwaway location, so the originals are
never modified, and records each notebook's status and duration.

Usage:
    uv run python scripts/run_notebooks.py [--root notebooks] [--timeout 600]
        [--kernel python3] [--pattern 'ch-*'] [--json results.json]

Exits non-zero if any notebook fails or times out.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_STATUS_LABELS = {"ok": "PASS", "failed": "FAIL", "timeout": "TIME"}
_RUNNER_KERNEL = "notebook-runner"


@dataclass
class NotebookResult:
    """Outcome of executing a single notebook."""

    path: Path
    status: str  # "ok" | "failed" | "timeout"
    duration_s: float
    error: str | None = None


def find_notebooks(root: Path, pattern: str = "ch-*") -> list[Path]:
    """Return sorted .ipynb files in directories matching pattern under root."""
    return sorted(
        nb
        for nb in root.glob(f"{pattern}/*.ipynb")
        if ".ipynb_checkpoints" not in nb.parts
    )


def build_execute_command(
    notebook: Path, output_dir: Path, timeout: int, kernel: str = "python3"
) -> list[str]:
    """Build the jupyter nbconvert command that executes a notebook."""
    return [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        "--output-dir",
        str(output_dir),
        str(notebook),
    ]


def _error_snippet(text: str, limit: int = 800) -> str:
    """Return the trimmed tail of a captured error stream."""
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def provision_kernel(dest: Path, name: str = _RUNNER_KERNEL) -> str:
    """Install a kernelspec bound to the current interpreter under dest.

    The default ``python3`` kernelspec may point at a bare ``python`` that
    resolves to the wrong interpreter. Binding a kernel to ``sys.executable``
    guarantees notebooks run in the same environment as this script (e.g. the
    project's uv venv, where the book's packages are installed).
    """
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--prefix", str(dest), "--name", name],
        check=True,
        capture_output=True,
    )
    return name


def run_notebook(
    notebook: Path,
    timeout: int = 600,
    kernel: str = "python3",
    jupyter_path: Path | None = None,
) -> NotebookResult:
    """Execute one notebook and return its result without modifying the original.

    If jupyter_path is given it is prepended to JUPYTER_PATH so a provisioned
    kernelspec located there is discoverable.
    """
    env = dict(os.environ)
    if jupyter_path is not None:
        existing = env.get("JUPYTER_PATH", "")
        env["JUPYTER_PATH"] = f"{jupyter_path}{os.pathsep}{existing}" if existing else str(jupyter_path)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = build_execute_command(notebook, Path(tmp), timeout, kernel)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout + 60, env=env
            )
        except subprocess.TimeoutExpired:
            return NotebookResult(notebook, "timeout", time.monotonic() - start, f"exceeded {timeout}s")
        duration = time.monotonic() - start
    if proc.returncode == 0:
        return NotebookResult(notebook, "ok", duration, None)
    return NotebookResult(notebook, "failed", duration, _error_snippet(proc.stderr))


def summarize(results: list[NotebookResult]) -> str:
    """Render a human-readable summary of notebook execution results."""
    lines = [
        f"  {r.status:<7} {r.path}  ({r.duration_s:.1f}s)" for r in results
    ]
    failed = [r for r in results if r.status != "ok"]
    passed = len(results) - len(failed)
    lines.append("")
    lines.append(f"{passed}/{len(results)} succeeded, {len(failed)} failed")
    if failed:
        lines.append("Failures:")
        lines.extend(f"  - {r.path} ({r.status})" for r in failed)
    return "\n".join(lines)


def _results_payload(results: list[NotebookResult]) -> list[dict]:
    """Convert results to JSON-serializable dicts."""
    payload = []
    for r in results:
        row = asdict(r)
        row["path"] = str(r.path)
        payload.append(row)
    return payload


def main() -> None:
    """CLI entry point: run all chapter notebooks and report results."""
    parser = argparse.ArgumentParser(description="Execute chapter notebooks and report pass/fail")
    parser.add_argument("--root", default="notebooks", type=Path)
    parser.add_argument("--pattern", default="ch-*")
    parser.add_argument("--timeout", default=600, type=int, help="per-notebook timeout (s)")
    parser.add_argument(
        "--kernel",
        default="auto",
        help="kernel name, or 'auto' (default) to run in this interpreter's environment",
    )
    parser.add_argument("--json", type=Path, help="optional path to write results as JSON")
    args = parser.parse_args()

    notebooks = find_notebooks(args.root, args.pattern)
    if not notebooks:
        print(f"No notebooks found under {args.root}/{args.pattern}/")
        return

    with tempfile.TemporaryDirectory() as kernel_home:
        if args.kernel == "auto":
            kernel = provision_kernel(Path(kernel_home))
            jupyter_path: Path | None = Path(kernel_home) / "share" / "jupyter"
            print(f"Running notebooks in this environment ({sys.executable})\n")
        else:
            kernel = args.kernel
            jupyter_path = None

        results: list[NotebookResult] = []
        for i, nb in enumerate(notebooks, 1):
            print(f"[{i}/{len(notebooks)}] {nb} ...", flush=True)
            result = run_notebook(nb, timeout=args.timeout, kernel=kernel, jupyter_path=jupyter_path)
            print(f"    {_STATUS_LABELS[result.status]} ({result.duration_s:.1f}s)", flush=True)
            results.append(result)

    print("\n" + summarize(results))
    if args.json:
        args.json.write_text(json.dumps(_results_payload(results), indent=2))
        print(f"\nWrote results to {args.json}")

    sys.exit(1 if any(r.status != "ok" for r in results) else 0)


if __name__ == "__main__":
    main()
