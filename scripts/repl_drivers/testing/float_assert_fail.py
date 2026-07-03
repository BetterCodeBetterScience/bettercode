#!/usr/bin/env python3
"""REPL driver: float_assert_fail.

Renders the transcript shown in book-testing.tex around the floating-point
assertion failure section (block 6, lines 160-163).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repl_console import Session


def main() -> None:
    s = Session()
    s.run("assert 0.1 + 0.2 == 0.3")
    sys.stdout.write(s.render())


if __name__ == "__main__":
    main()
