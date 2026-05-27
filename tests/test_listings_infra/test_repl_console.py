"""Tests for the REPL Session helper."""

from repl_console import Session


def test_simple_expression_renders_with_prompt_and_result():
    s = Session()
    s.run("1 + 1")
    assert s.render() == ">>> 1 + 1\n2\n"


def test_assignment_produces_no_output():
    s = Session()
    s.run("x = 5")
    assert s.render() == ">>> x = 5\n"


def test_state_persists_across_run_calls():
    s = Session()
    s.run("x = 5")
    s.run("x + 1")
    assert s.render() == ">>> x = 5\n>>> x + 1\n6\n"


def test_print_output_is_captured():
    s = Session()
    s.run("print('hello')")
    assert s.render() == ">>> print('hello')\nhello\n"


def test_exception_renders_traceback():
    s = Session()
    s.run("1 / 0")
    out = s.render()
    assert ">>> 1 / 0\n" in out
    assert "Traceback (most recent call last):" in out
    assert "ZeroDivisionError: division by zero" in out


def test_multi_line_input_renders_continuation_prompts():
    s = Session()
    s.run("def f(x):\n    return x * 2")
    s.run("f(3)")
    out = s.render()
    assert ">>> def f(x):\n... " in out
    assert "    return x * 2" in out
    assert ">>> f(3)\n6\n" in out


def test_import_persists_across_runs():
    s = Session()
    s.run("import math")
    s.run("math.sqrt(16)")
    assert ">>> math.sqrt(16)\n4.0\n" in s.render()
