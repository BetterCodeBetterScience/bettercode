#!/usr/bin/env python3
"""REPL driver: find_outliers_zero_div.

Renders the transcript shown in book-testing.tex showing a ZeroDivisionError
when all data values are equal (block 13, lines 278-293).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repl_console import Session


def main() -> None:
    s = Session()
    s.run("from bettercode.testing.find_outliers_v1_buggy import find_outliers")
    s.run("data = [1, 1, 1, 1, 1]")
    s.run("find_outliers(data)")
    sys.stdout.write(s.render())


if __name__ == "__main__":
    main()
