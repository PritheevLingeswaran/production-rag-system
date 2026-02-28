from __future__ import annotations

from pathlib import Path


def load_prompt(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing prompt: {path}")
    return p.read_text(encoding="utf-8")
