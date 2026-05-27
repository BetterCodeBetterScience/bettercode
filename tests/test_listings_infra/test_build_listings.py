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
