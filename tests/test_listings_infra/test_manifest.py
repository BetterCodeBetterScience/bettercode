"""Tests for the listings-manifest loader."""

import pytest
from manifest import load_manifest


def test_load_manifest_parses_command_entry(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: pytest_pass
    description: "Passing pytest"
    command: ["pytest", "tests/foo.py"]
    output_file: "latex/generated/testing/pytest_pass.txt"
    expected_exit_code: 0
    normalize: [pytest_timings]
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    m = load_manifest(p)
    assert m.chapter == "testing"
    assert len(m.outputs) == 1
    entry = m.outputs[0]
    assert entry.id == "pytest_pass"
    assert entry.command == ["pytest", "tests/foo.py"]
    assert entry.driver is None
    assert entry.output_file == "latex/generated/testing/pytest_pass.txt"
    assert entry.expected_exit_code == 0
    assert entry.normalize == ["pytest_timings"]


def test_load_manifest_parses_driver_entry(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: repl_one
    description: "REPL one"
    driver: "scripts/repl_drivers/testing/one.py"
    output_file: "latex/generated/testing/repl_one.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    m = load_manifest(p)
    entry = m.outputs[0]
    assert entry.driver == "scripts/repl_drivers/testing/one.py"
    assert entry.command is None


def test_load_manifest_rejects_both_command_and_driver(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: bad
    description: "Bad"
    command: ["true"]
    driver: "scripts/x.py"
    output_file: "latex/generated/testing/bad.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="exactly one of"):
        load_manifest(p)


def test_load_manifest_rejects_neither_command_nor_driver(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: bad
    description: "Bad"
    output_file: "latex/generated/testing/bad.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="exactly one of"):
        load_manifest(p)


def test_load_manifest_rejects_duplicate_ids(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: same
    description: "A"
    command: ["true"]
    output_file: "latex/generated/testing/a.txt"
    expected_exit_code: 0
    normalize: []
  - id: same
    description: "B"
    command: ["true"]
    output_file: "latex/generated/testing/b.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)


def test_load_manifest_defaults_cwd_to_dot(tmp_path):
    yaml_text = """
chapter: testing
outputs:
  - id: x
    description: "X"
    command: ["true"]
    output_file: "latex/generated/testing/x.txt"
    expected_exit_code: 0
    normalize: []
"""
    p = tmp_path / "testing.yaml"
    p.write_text(yaml_text)
    m = load_manifest(p)
    assert m.outputs[0].cwd == "."
