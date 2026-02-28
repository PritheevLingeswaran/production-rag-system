# Changelog

## [0.1.0] - 2026-02-08
- Strict-grounding RAG system:
  - PDF/TXT ingestion → token-aware chunking → embeddings → vector index
  - Vector store: Chroma (default), optional FAISS, extension point for Pinecone
  - FastAPI `/query` API with citations + confidence + cost metrics
  - Evaluation harness (retrieval/answer/hallucination/calibration/cost)
  - Structured logging + Prometheus metrics
  - Docker + CI
