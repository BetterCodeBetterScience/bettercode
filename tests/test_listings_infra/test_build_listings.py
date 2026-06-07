"""End-to-end tests for the build_listings CLI."""

import subprocess
import sys
from pathlib import Path

import pytest

import build_listings

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
    for name in (
        "build_listings.py",
        "manifest.py",
        "normalize_output.py",
        "repl_console.py",
    ):
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
        '    command: ["python", "-c", "print(\'hello in 0.5s\')"]\n'
        '    output_file: "latex/generated/testing/echo.txt"\n'
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
        '    command: ["python", "-c", "import sys; sys.exit(2)"]\n'
        '    output_file: "latex/generated/testing/must_pass.txt"\n'
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
        '    command: ["python", "-c", "import sys; print(\'boom\'); sys.exit(1)"]\n'
        '    output_file: "latex/generated/testing/expect_fail.txt"\n'
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
        '    command: ["python", "-c", "print(\'A\')"]\n'
        '    output_file: "latex/generated/testing/a.txt"\n'
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
        "  - id: b\n"
        "    description: b\n"
        '    command: ["python", "-c", "print(\'B\')"]\n'
        '    output_file: "latex/generated/testing/b.txt"\n'
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    result = _run_cli(["build", "testing", "--only", "b"], cwd=tmp_repo)
    assert result.returncode == 0, result.stderr
    assert not (tmp_repo / "latex" / "generated" / "testing" / "a.txt").exists()
    assert (tmp_repo / "latex" / "generated" / "testing" / "b.txt").read_text() == "B\n"


def test_verify_passes_when_committed_matches(tmp_repo):
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: e\n"
        "    description: e\n"
        '    command: ["python", "-c", "print(\'E\')"]\n'
        '    output_file: "latex/generated/testing/e.txt"\n'
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
        '    command: ["python", "-c", "print(\'NEW\')"]\n'
        '    output_file: "latex/generated/testing/e.txt"\n'
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
        '    command: ["python", "-c", "print(\'X\')"]\n'
        '    output_file: "latex/generated/testing/missing.txt"\n'
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    result = _run_cli(["verify", "testing"], cwd=tmp_repo)
    assert result.returncode != 0
    assert "missing" in (result.stdout + result.stderr).lower()


def test_verify_ignores_pytest_timing_only_drift(tmp_repo):
    """Verify should pass when only pytest-style timing values differ.

    Each run regenerates output with the current wall-clock time. We
    don't want that natural variation to count as drift.
    """
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: timed\n"
        "    description: timed pytest-style output\n"
        '    command: ["python", "-c", "print(\'1 passed in 0.20s\')"]\n'
        '    output_file: "latex/generated/testing/timed.txt"\n'
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    # Committed output has a different timing; everything else identical.
    (tmp_repo / "latex" / "generated" / "testing" / "timed.txt").write_text(
        "1 passed in 0.14s\n"
    )
    result = _run_cli(["verify", "testing"], cwd=tmp_repo)
    assert result.returncode == 0, (
        f"expected verify to ignore timing-only drift; stderr:\n{result.stderr}"
    )


def test_command_prompt_strips_uv_run_wrapper():
    """A pytest command renders as `$ pytest ...` with the uv wrapper dropped."""
    cmd = ["uv", "run", "pytest", "--tb=short", "tests/foo.py"]
    assert build_listings._command_prompt(cmd) == "$ pytest --tb=short tests/foo.py"


def test_command_prompt_quotes_args_with_spaces():
    """Arguments containing spaces are shell-quoted."""
    cmd = ["uv", "run", "pytest", "-m", "not unit"]
    assert build_listings._command_prompt(cmd) == "$ pytest -m 'not unit'"


def test_command_prompt_returns_none_for_non_pytest():
    """Non-pytest commands get no prompt line (output is left untouched)."""
    assert build_listings._command_prompt(["python", "-c", "print('x')"]) is None


def test_build_prepends_pytest_command_line(tmp_repo):
    """Generated output for a pytest entry starts with the `$ pytest ...` line."""
    test_file = tmp_repo / "test_sample.py"
    test_file.write_text("def test_ok():\n    assert True\n")
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: pytest_sample\n"
        "    description: a real pytest run\n"
        '    command: ["python", "-m", "pytest", "--no-header", "-q", "test_sample.py"]\n'
        '    output_file: "latex/generated/testing/pytest_sample.txt"\n'
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    result = _run_cli(["build", "testing"], cwd=tmp_repo)
    assert result.returncode == 0, result.stderr
    out_file = tmp_repo / "latex" / "generated" / "testing" / "pytest_sample.txt"
    lines = out_file.read_text().splitlines()
    assert lines[0] == "$ pytest --no-header -q test_sample.py"


def test_build_strips_testing_chapter_dir_from_command_and_output(tmp_repo):
    """The strip_testing_chapter_dir normalizer cleans the prompt line too."""
    test_dir = tmp_repo / "tests" / "testing_chapter"
    test_dir.mkdir(parents=True)
    (test_dir / "test_sample.py").write_text("def test_ok():\n    assert True\n")
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: pytest_sample\n"
        "    description: a real pytest run\n"
        '    command: ["python", "-m", "pytest", "--no-header", "-v",'
        ' "tests/testing_chapter/test_sample.py"]\n'
        '    output_file: "latex/generated/testing/pytest_sample.txt"\n'
        "    expected_exit_code: 0\n"
        "    normalize: [strip_testing_chapter_dir]\n"
    )
    result = _run_cli(["build", "testing"], cwd=tmp_repo)
    assert result.returncode == 0, result.stderr
    text = (tmp_repo / "latex" / "generated" / "testing" / "pytest_sample.txt").read_text()
    assert "tests/testing_chapter/" not in text
    assert text.splitlines()[0] == "$ pytest --no-header -v test_sample.py"


def test_verify_still_fails_on_substantive_drift_with_timings(tmp_repo):
    """Verify should still fail when non-timing content differs."""
    manifest = tmp_repo / "latex" / "manifests" / "testing.yaml"
    manifest.write_text(
        "chapter: testing\n"
        "outputs:\n"
        "  - id: timed\n"
        "    description: timed pytest-style output\n"
        '    command: ["python", "-c", "print(\'2 passed in 0.20s\')"]\n'
        '    output_file: "latex/generated/testing/timed.txt"\n'
        "    expected_exit_code: 0\n"
        "    normalize: []\n"
    )
    # Committed says "1 passed", regenerated says "2 passed" — substantive.
    (tmp_repo / "latex" / "generated" / "testing" / "timed.txt").write_text(
        "1 passed in 0.14s\n"
    )
    result = _run_cli(["verify", "testing"], cwd=tmp_repo)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "DRIFT" in output
