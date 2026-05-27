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
