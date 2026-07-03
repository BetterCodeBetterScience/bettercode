#!/usr/bin/env python3
"""REPL driver: float_allclose.

Renders the transcript shown in book-testing.tex showing numpy allclose
usage for floating-point comparison (block 8, lines 176-179).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repl_console import Session


def main() -> None:
    s = Session()
    s.run("import numpy as np")
    s.run("np.allclose(0.1 + 0.2, 0.3)")
    sys.stdout.write(s.render())


if __name__ == "__main__":
    main()
