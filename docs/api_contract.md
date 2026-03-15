# API Contract

## Design rules

- Product-facing routes are versioned under `/api/v1`.
- Compatibility endpoints still exist for the older surface.
- Error responses use a structured envelope:

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request validation failed.",
    "request_id": "3e1c3c34-7c70-4a1d-b2b4-7df7c3b1cb20",
    "details": {
      "errors": []
    }
  }
}
```

## `POST /api/v1/query`

Request:

```json
{
  "query": "How many projects are there in the resume?",
  "top_k": 8,
  "rewrite_query": false,
  "filter": {
    "source": "resume"
  }
}
```

Response:

```json
{
  "answer": "3 [resume.txt:p1:c0]",
  "confidence": 0.65,
  "sources": [
    {
      "chunk_id": "resume.txt:p1:c0",
      "source": "data/raw/documents/uploads/local-user/resume.txt",
      "page": 1,
      "score": 1.0,
      "text": "Production-Grade Hybrid RAG System..."
    }
  ],
  "refusal": {
    "is_refusal": false,
    "reason": ""
  },
  "metrics": {
    "latency_ms": 18.1,
    "retrieval_latency_ms": 5.2,
    "generation_latency_ms": 12.9,
    "embedding_tokens": 0,
    "llm_tokens_in": 35,
    "llm_tokens_out": 8,
    "citation_coverage": 1.0
  }
}
```

## `POST /api/v1/documents/upload`

Multipart form request:

- field: `files`
- optional field: `collection_name`

Response:

```json
{
  "documents": [
    {
      "id": "doc-1",
      "filename": "resume.txt",
      "stored_path": "data/raw/documents/uploads/local-user/uuid-resume.txt",
      "file_type": "txt",
      "size_bytes": 431,
      "indexing_status": "queued",
      "summary_status": "queued",
      "upload_time": "2026-03-15T09:12:13Z"
    }
  ]
}
```

## `GET /api/v1/documents`

Query params:

- `search`
- `sort`
- `order`

Response:

```json
{
  "documents": [
    {
      "id": "doc-1",
      "filename": "resume.txt",
      "stored_path": "data/raw/documents/uploads/local-user/uuid-resume.txt",
      "file_type": "txt",
      "size_bytes": 431,
      "pages": 1,
      "chunks_created": 2,
      "upload_time": "2026-03-15T09:12:13Z",
      "indexing_status": "ready",
      "summary_status": "ready",
      "collection_name": null,
      "error_message": null,
      "metadata": {}
    }
  ]
}
```

## `POST /api/v1/chat/query`

Request:

```json
{
  "question": "How many projects are there in the resume?",
  "session_id": null,
  "retrieval_mode": "hybrid_rrf",
  "top_k": 5
}
```

Response:

```json
{
  "session_id": "session-1",
  "answer": "3 [resume.txt:p1:c0]",
  "confidence": 0.65,
  "refusal": {
    "is_refusal": false,
    "reason": ""
  },
  "citations": [
    {
      "id": "citation-1",
      "document_id": "doc-1",
      "chunk_id": "resume.txt:p1:c0",
      "source": "resume.txt",
      "page": 1,
      "excerpt": "Production-Grade Hybrid RAG System...",
      "score": 1.0,
      "created_at": "2026-03-15T09:12:13Z"
    }
  ],
  "sources": [
    {
      "chunk_id": "resume.txt:p1:c0",
      "source": "resume.txt",
      "page": 1,
      "score": 1.0,
      "text": "Production-Grade Hybrid RAG System..."
    }
  ],
  "timing": {
    "total_latency_ms": 18.1,
    "retrieval_latency_ms": 5.2,
    "generation_latency_ms": 12.9,
    "embedding_tokens": 0,
    "llm_tokens_in": 35,
    "llm_tokens_out": 8,
    "llm_cost_usd": null
  }
}
```

## `GET /healthz` and `GET /readiness`

Healthy response:

```json
{
  "status": "ok",
  "service": "rag-smart-qa-api",
  "environment": "prod",
  "checks": {
    "runtime": "ok",
    "documents": "ok"
  }
}
```

## `GET /metrics`

- Content type: Prometheus text exposition format
- Also available at `/api/v1/metrics`
