# rag-smart-qa

Industry-ready, **strict-grounding** Retrieval-Augmented Generation (RAG) system for PDF/TXT Q&A.

Core guarantees:
- answers are generated **only from retrieved chunks**
- **citations** are mandatory; missing/invalid citations → refusal
- per-request **confidence** + token/cost accounting
- evaluation harness for retrieval + answer quality + hallucination rate + calibration

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
make install

cp .env.example .env
# set OPENAI_API_KEY (or use local embeddings model)
```

Put docs in:
- `data/raw/documents/`

Ingest + index:
```bash
make ingest
make index
```

Run API:
```bash
make run
```

Query:
```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the warranty period?", "top_k": 6}'
```

## Vector stores
Default: **Chroma** (easy local persistence + metadata).  
Optional: **FAISS** (faster, but requires installing `faiss-cpu` and compatible CPU).

To enable FAISS:
- install `faiss-cpu`
- set `vector_store.provider: faiss` in `configs/base.yaml`

## Evaluation
Put gold dataset at `evaluation/datasets/gold.jsonl` and run:
```bash
make eval
```

The evaluation report (`docs/evaluation_results.md`) includes:
- recall@k / precision@k (dense vs hybrid)
- hallucination rate
- p95 latency (retrieval, generation, end-to-end) for the local eval run
- corpus size (chunks, unique sources)

## Load test (HTTP)
To measure p95 latency and throughput under concurrency:

1) Start the API (`make run`)
2) In another terminal:
```bash
make loadtest
```

Outputs:
- `docs/load_test_results.md`
- `docs/load_test_results.json`

See docs:
- `docs/architecture.md`
- `docs/decisions.md`
- `docs/tradeoffs.md`
- `docs/security.md`
