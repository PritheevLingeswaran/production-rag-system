from __future__ import annotations

from utils.text import normalize_whitespace, strip_null_bytes
from utils.settings import CleaningConfig


def clean_text(text: str, cfg: CleaningConfig) -> str:
    if cfg.drop_null_bytes:
        text = strip_null_bytes(text)
    if cfg.normalize_whitespace:
        text = normalize_whitespace(text)
    return text
