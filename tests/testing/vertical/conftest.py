"""Put the vertical library dir on ``sys.path`` so the tests import it directly."""

import sys
from pathlib import Path

VERTICAL_MODULES_DIR = Path(__file__).resolve().parents[3] / "testing" / "vertical"
sys.path.insert(0, str(VERTICAL_MODULES_DIR))
