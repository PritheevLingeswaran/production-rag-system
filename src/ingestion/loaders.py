from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str
    page: int
    text: str


def load_pdf(path: Path) -> List[Page]:
    reader = PdfReader(str(path))
    pages: List[Page] = []
    for i, p in enumerate(reader.pages):
        txt = p.extract_text() or ""
        pages.append(Page(source=str(path), page=i + 1, text=txt))
    return pages


def load_txt(path: Path) -> List[Page]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return [Page(source=str(path), page=1, text=txt)]


def iter_documents(raw_dir: str, exts: List[str]) -> Iterator[Path]:
    root = Path(raw_dir)
    if not root.exists():
        return
    exts_l = {e.lower() for e in exts}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_l:
            yield p
