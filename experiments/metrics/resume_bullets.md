# Resume Bullets

## Short Bullet
- Built a hybrid RAG system with reproducible evaluation (2 documents / 9 chunks; 8.058 ms p95 hybrid retrieval latency).

## Strong Bullet
- Implemented a reproducible dense-vs-hybrid RAG evaluation pipeline: hallucination rate changed by -0.2727, hybrid p95 latency 8.058 ms.

## Senior/Staff-Style Bullet
- Productionized measurement for a hybrid RAG stack with code-generated resume metrics, covering corpus size, dense-vs-hybrid retrieval quality, and offline hallucination checks (2 documents / 9 chunks; 8.058 ms p95 hybrid retrieval latency).

## Claim Boundaries
- These bullets include only values written into `experiments/metrics/resume_metrics.json` by code execution.
- No throughput or RPS claim is included because resume metrics omit it unless explicitly measured and selected.
