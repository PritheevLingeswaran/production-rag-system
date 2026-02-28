"""Auto-add ./src to sys.path for local execution.

This keeps imports stable whether you run via `pip install -e .` or directly from the repo.
"""

import sys
from pathlib import Path

src = Path(__file__).resolve().parent / "src"
if src.exists():
    sys.path.insert(0, str(src))
