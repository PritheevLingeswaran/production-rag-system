# Data directory

- `raw/documents/`: put source documents here (PDF/TXT)
- `processed/` is generated artifacts:
  - `chunks/`: extracted + chunked text (JSONL)
  - `indexes/`: vector index persistence (Chroma/FAISS)

Do not commit `processed/` in real deployments.
