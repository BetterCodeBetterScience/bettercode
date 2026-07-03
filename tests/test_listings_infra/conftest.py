"""Test config for the listings-build infrastructure.

Adds the `scripts/` directory to sys.path so tests can import the
script modules directly (they are not installed as a package).
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
