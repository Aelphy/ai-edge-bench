#!/usr/bin/env python3
"""Host/container CLI for Level 2 (`layer/catalog/`) validation & scouting."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "model" / "docker" / "harness"
if str(HARNESS_DIR) not in sys.path:
  sys.path.insert(0, str(HARNESS_DIR))

# pylint: disable=g-import-not-at-top,wrong-import-position
from aeb import catalog

# pylint: enable=g-import-not-at-top,wrong-import-position


if __name__ == "__main__":
  sys.exit(catalog.main())
