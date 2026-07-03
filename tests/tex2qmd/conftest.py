"""Test config for tex2qmd.

Adds the `scripts/` directory to sys.path so tests can import the
tex2qmd modules directly (they are not installed as a package).
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
