"""Tests for output normalizers."""

from normalize_output import apply_normalizers, NORMALIZERS


def test_pytest_timings_replaces_decimal_seconds():
    text = "===== 1 passed in 0.10s ====="
    result = NORMALIZERS["pytest_timings"](text)
    assert result == "===== 1 passed in X.XXs ====="


def test_pytest_timings_handles_multiple_occurrences():
    text = "1 failed, 2 passed in 1.23s\nshort test summary: 0.45s\n"
    result = NORMALIZERS["pytest_timings"](text)
    assert "X.XXs" in result
    assert "1.23s" not in result
    assert "0.45s" not in result


def test_pytest_timings_leaves_non_timing_text_untouched():
    text = "tests/test_foo.py::test_bar PASSED"
    assert NORMALIZERS["pytest_timings"](text) == text


def test_strip_testing_chapter_dir_removes_prefix_everywhere():
    """The tests/testing_chapter/ prefix is removed from every occurrence."""
    text = (
        "$ pytest --no-header tests/testing_chapter/test_distance.py\n"
        "tests/testing_chapter/test_distance.py::test_distance PASSED\n"
    )
    result = NORMALIZERS["strip_testing_chapter_dir"](text)
    assert result == (
        "$ pytest --no-header test_distance.py\n"
        "test_distance.py::test_distance PASSED\n"
    )


def test_strip_testing_chapter_dir_leaves_other_text_untouched():
    text = "tests/other/test_foo.py::test_bar PASSED"
    assert NORMALIZERS["strip_testing_chapter_dir"](text) == text


def test_memory_addresses_collapses_hex_pointers():
    text = "<function allclose at 0x101403370>"
    result = NORMALIZERS["memory_addresses"](text)
    assert result == "<function allclose at 0xADDR>"


def test_memory_addresses_only_hits_0x_prefix():
    text = "expected 100, got 200"
    assert NORMALIZERS["memory_addresses"](text) == text


def test_file_paths_strips_repo_absolute_to_relative(tmp_path, monkeypatch):
    """Absolute paths under the repo root are rewritten to repo-relative."""
    monkeypatch.chdir(tmp_path)
    repo_root = str(tmp_path)
    text = f'  File "{repo_root}/src/bettercode/foo.py", line 12, in foo'
    result = NORMALIZERS["file_paths"](text)
    assert result == '  File "src/bettercode/foo.py", line 12, in foo'


def test_file_paths_leaves_non_repo_paths_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = '  File "/usr/lib/python3.12/site-packages/x.py", line 1, in y'
    assert NORMALIZERS["file_paths"](text) == text


def test_file_paths_replaces_standalone_cwd_with_dot(tmp_path, monkeypatch):
    """pytest's `rootdir:` line shows cwd without a trailing slash."""
    monkeypatch.chdir(tmp_path)
    text = f"rootdir: {tmp_path}\nconfigfile: pyproject.toml"
    result = NORMALIZERS["file_paths"](text)
    assert result == "rootdir: .\nconfigfile: pyproject.toml"


def test_temp_paths_normalizes_macos_temp():
    text = "wrote /var/folders/aa/bb/T/tmpXYZ.txt"
    result = NORMALIZERS["temp_paths"](text)
    assert result == "wrote /tmp/PATH.txt"


def test_temp_paths_normalizes_generic_tmp():
    text = "opened /tmp/pytest-of-user/pytest-12/abc.txt"
    result = NORMALIZERS["temp_paths"](text)
    assert "pytest-of-user" not in result
    assert "/tmp/PATH" in result


def test_terminal_width_truncates_long_rule_lines():
    text = "=" * 120 + "\nactual content\n" + "=" * 90
    result = NORMALIZERS["terminal_width"](text)
    lines = result.split("\n")
    assert lines[0] == "=" * 64
    assert lines[1] == "actual content"
    assert lines[2] == "=" * 64


def test_terminal_width_leaves_short_rules_alone():
    text = "==== short rule ====\n"
    assert NORMALIZERS["terminal_width"](text) == text


def test_apply_normalizers_runs_in_order():
    text = "function at 0x123 ran in 0.5s"
    result = apply_normalizers(text, ["memory_addresses", "pytest_timings"])
    assert result == "function at 0xADDR ran in X.XXs"


def test_apply_normalizers_unknown_name_raises():
    import pytest

    with pytest.raises(KeyError):
        apply_normalizers("anything", ["does_not_exist"])


def test_worker_count_normalizes_xdist_worker_line():
    text = "scheduling tests via LoadScheduling\n16 workers [10 items]"
    result = NORMALIZERS["worker_count"](text)
    assert result == "scheduling tests via LoadScheduling\nN workers [10 items]"


def test_worker_count_leaves_other_numbers_alone():
    text = "16 tests passed"
    assert NORMALIZERS["worker_count"](text) == text


def test_strip_docstrings_removes_multi_line_docstring():
    text = (
        '    def test_foo():\n'
        '        """\n'
        '        Multi-line docstring.\n'
        '        Second line of docstring.\n'
        '        """\n'
        '        x = 1\n'
    )
    expected = '    def test_foo():\n        x = 1\n'
    assert NORMALIZERS["strip_docstrings"](text) == expected


def test_strip_docstrings_removes_single_line_docstring():
    text = (
        '    def test_foo():\n'
        '        """Single line docstring."""\n'
        '        x = 1\n'
    )
    expected = '    def test_foo():\n        x = 1\n'
    assert NORMALIZERS["strip_docstrings"](text) == expected


def test_strip_docstrings_leaves_string_literals_alone():
    """A `\"\"\"` inside a code expression isn't at column 0 of whitespace-only."""
    text = '    msg = "hello"\n    print(msg)\n'
    assert NORMALIZERS["strip_docstrings"](text) == text


def test_strip_docstrings_handles_multiple_docstrings():
    text = (
        '    def a():\n'
        '        """doc a"""\n'
        '        return 1\n'
        '    def b():\n'
        '        """doc b"""\n'
        '        return 2\n'
    )
    expected = (
        '    def a():\n'
        '        return 1\n'
        '    def b():\n'
        '        return 2\n'
    )
    assert NORMALIZERS["strip_docstrings"](text) == expected


# --- xdist_schedule normalizer tests ---

XDIST_SAMPLE_A = (
    "============================= test session starts ==============================\n"
    "scheduling tests via LoadScheduling\n"
    "\n"
    "tests/t.py::test_a[0] \n"
    "tests/t.py::test_a[2] \n"
    "tests/t.py::test_a[1] \n"
    "[gw4] [ 10%] PASSED tests/t.py::test_a[0] \n"
    "[gw1] [ 20%] PASSED tests/t.py::test_a[2] \n"
    "[gw2] [ 30%] PASSED tests/t.py::test_a[1] \n"
    "\n"
    "============================== 3 passed in 1.00s ==============================\n"
)

XDIST_SAMPLE_B = (
    "============================= test session starts ==============================\n"
    "scheduling tests via LoadScheduling\n"
    "\n"
    "tests/t.py::test_a[1] \n"
    "tests/t.py::test_a[0] \n"
    "tests/t.py::test_a[2] \n"
    "[gw2] [ 10%] PASSED tests/t.py::test_a[2] \n"
    "[gw0] [ 20%] PASSED tests/t.py::test_a[1] \n"
    "[gw9] [ 30%] PASSED tests/t.py::test_a[0] \n"
    "\n"
    "============================== 3 passed in 1.00s ==============================\n"
)


def test_xdist_schedule_is_idempotent():
    """Applying the normalizer twice gives the same result as applying once."""
    result = NORMALIZERS["xdist_schedule"](XDIST_SAMPLE_A)
    assert NORMALIZERS["xdist_schedule"](result) == result


def test_xdist_schedule_produces_same_output_for_different_orderings():
    """Two runs with different worker/test orderings produce identical output."""
    result_a = NORMALIZERS["xdist_schedule"](XDIST_SAMPLE_A)
    result_b = NORMALIZERS["xdist_schedule"](XDIST_SAMPLE_B)
    assert result_a == result_b


def test_xdist_schedule_strips_worker_ids():
    """Worker IDs like [gw4] are replaced with [gwN]."""
    result = NORMALIZERS["xdist_schedule"](XDIST_SAMPLE_A)
    assert "[gw4]" not in result
    assert "[gw1]" not in result
    assert "[gwN]" in result


def test_xdist_schedule_preserves_summary_line():
    """The final summary line (passed/failed count) is preserved."""
    result = NORMALIZERS["xdist_schedule"](XDIST_SAMPLE_A)
    assert "3 passed" in result


def test_xdist_schedule_leaves_unrelated_text_unchanged():
    """Text without xdist scheduling markers is returned unchanged."""
    text = "plain pytest output\n1 passed in 0.1s\n"
    assert NORMALIZERS["xdist_schedule"](text) == text
