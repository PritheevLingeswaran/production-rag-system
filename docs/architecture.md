# Architecture

This system implements a **strict-grounding RAG pipeline**:

1. **Ingestion**: PDF/TXT → page text + metadata (source, page)
2. **Preprocessing**: cleaning → token-aware chunking (chunk_size + overlap)
3. **Indexing**: embed chunks → write to vector store (Chroma default; FAISS optional)
4. **Query time**:
   - optional query rewrite to improve recall
   - embed query
   - retrieve top-k chunks (+ optional hybrid/rerank)
   - grounded generation with mandatory citations
   - refusal if evidence is insufficient

## Data flow (Mermaid)

```mermaid
flowchart LR
  A[Docs<br/>PDF/TXT] --> B[Ingestion]
  B --> C[Preprocess<br/>clean + chunk]
  C --> D[Embed chunks]
  D --> E[Vector store<br/>Chroma/FAISS]
  Q[Question] --> R[Rewrite query]
  R --> QE[Embed query]
  QE --> S[Retrieve top-k]
  E --> S
  S --> G[Generate answer<br/>strict citations]
  G --> O[API Response<br/>answer + sources + confidence]
  G --> M[Metrics<br/>tokens/cost/latency]
```
