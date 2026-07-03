"""Tests for the chapter-notebook execution runner (scripts/run_notebooks.py)."""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_notebooks import (
    NotebookResult,
    build_execute_command,
    find_notebooks,
    provision_kernel,
    run_notebook,
    summarize,
)


def _write_notebook(path, source):
    """Write a minimal single-code-cell notebook running source at path."""
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb))


def test_find_notebooks_globs_chapter_dirs(tmp_path):
    """find_notebooks returns sorted ch-*/*.ipynb and ignores other dirs and files."""
    (tmp_path / "ch-05").mkdir()
    (tmp_path / "ch-10").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "ch-05" / "a.ipynb").write_text("{}")
    (tmp_path / "ch-10" / "b.ipynb").write_text("{}")
    (tmp_path / "ch-05" / "notes.md").write_text("x")
    (tmp_path / "other" / "c.ipynb").write_text("{}")
    found = find_notebooks(tmp_path)
    assert [p.name for p in found] == ["a.ipynb", "b.ipynb"]


def test_find_notebooks_skips_checkpoints(tmp_path):
    """Checkpoint notebooks are not returned."""
    (tmp_path / "ch-05" / ".ipynb_checkpoints").mkdir(parents=True)
    (tmp_path / "ch-05" / "a.ipynb").write_text("{}")
    (tmp_path / "ch-05" / ".ipynb_checkpoints" / "a-checkpoint.ipynb").write_text("{}")
    found = find_notebooks(tmp_path)
    assert [p.name for p in found] == ["a.ipynb"]


def test_build_execute_command_has_execute_timeout_and_path(tmp_path):
    """The command executes to a notebook with timeout, kernel, and the source path."""
    nb = tmp_path / "x.ipynb"
    cmd = build_execute_command(nb, output_dir=tmp_path, timeout=123, kernel="python3")
    assert "nbconvert" in cmd
    assert "--execute" in cmd
    assert "--to" in cmd and "notebook" in cmd
    assert any("123" in part for part in cmd)
    assert any("python3" in part for part in cmd)
    assert str(nb) in cmd


def test_summarize_reports_counts_and_failures():
    """The summary reports the pass count and lists failed notebooks by path."""
    results = [
        NotebookResult(Path("ch-05/a.ipynb"), "ok", 1.2, None),
        NotebookResult(Path("ch-09/b.ipynb"), "failed", 3.4, "ValueError: boom"),
    ]
    text = summarize(results)
    assert "1/2 succeeded" in text
    assert "ok" in text
    assert "failed" in text
    assert "ch-09/b.ipynb" in text


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("jupyter") is None, reason="jupyter not installed")
def test_run_notebook_success(tmp_path):
    """A notebook whose cell runs cleanly is recorded as ok."""
    nb = tmp_path / "ok.ipynb"
    _write_notebook(nb, "x = 1 + 1\nassert x == 2")
    result = run_notebook(nb, timeout=120)
    assert result.status == "ok"
    assert result.error is None


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("jupyter") is None, reason="jupyter not installed")
def test_provision_kernel_binds_current_interpreter(tmp_path):
    """The provisioned kernelspec launches the interpreter running the tests."""
    name = provision_kernel(tmp_path)
    kernel_json = tmp_path / "share" / "jupyter" / "kernels" / name / "kernel.json"
    argv0 = json.loads(kernel_json.read_text())["argv"][0]
    assert argv0 == sys.executable


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("jupyter") is None, reason="jupyter not installed")
def test_run_notebook_failure_records_error(tmp_path):
    """A notebook whose cell raises is recorded as failed with error text."""
    nb = tmp_path / "bad.ipynb"
    _write_notebook(nb, "raise ValueError('boom')")
    result = run_notebook(nb, timeout=120)
    assert result.status == "failed"
    assert result.error is not None
    assert "ValueError" in result.error or "boom" in result.error
