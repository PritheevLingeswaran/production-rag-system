from __future__ import annotations

from typing import List, Tuple

from ingestion.loaders import Page
from preprocessing.cleaning import clean_text
from preprocessing.chunking import chunk_text
from utils.settings import Settings


def preprocess_pages_to_chunks(settings: Settings, pages: List[Page]) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    for page in pages:
        cleaned = clean_text(page.text, settings.preprocessing.cleaning)
        for ch in chunk_text(cleaned, settings.preprocessing.chunking):
            out.append((page.page, ch.idx, ch.text))
    return out
