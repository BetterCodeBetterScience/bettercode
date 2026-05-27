#!/usr/bin/env python3
"""REPL driver: float_value.

Renders the transcript shown in book-testing.tex showing the floating-point
representation of 0.1 + 0.2 (block 7, lines 169-170).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repl_console import Session


def main() -> None:
    s = Session()
    s.run("0.1 + 0.2")
    sys.stdout.write(s.render())


if __name__ == "__main__":
    main()
