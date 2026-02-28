from __future__ import annotations

from ingestion.ingest import ingest_documents, write_chunks
from utils.config import ensure_dirs, load_settings
from utils.logging import configure_logging


def ingest_main() -> None:
    settings = load_settings()
    ensure_dirs(settings)
    configure_logging()
    chunks = ingest_documents(settings)
    write_chunks(settings, chunks)


if __name__ == "__main__":
    ingest_main()
