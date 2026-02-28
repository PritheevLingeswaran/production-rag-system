# rag-smart-qa

Industry-ready, strict-grounding Retrieval-Augmented Generation (RAG) system for PDF/TXT question answering.

---

## 🔒 Core Guarantees

- Answers are generated **only from retrieved document chunks**
- Citations are mandatory — missing or invalid citations trigger automatic refusal
- Per-request confidence scoring
- Token usage and cost tracking
- Hybrid retrieval (BM25 + Dense embeddings)
- Full evaluation harness:
  - Recall@k / Precision@k
  - Hallucination rate
  - Confidence calibration
  - P95 latency
  - Cost per query

---

## 🏗 Architecture Overview

Hybrid Retrieval Pipeline:

Query  
→ Dense Retrieval (Chroma / FAISS)  
→ BM25 Retrieval  
→ Score Fusion (configurable weight)  
→ Strict Grounded Prompt  
→ LLM Generation  
→ Citation Validation  
→ Response (Answer + Confidence + Metrics)

---

## 🖥 System Requirements

- Python 3.10+
- pip
- Internet (first run downloads embedding model)

Optional:
- OpenAI API Key (for LLM generation)
- FAISS (optional backend)

---

# 🚀 Setup (macOS / Linux)

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and set OPENAI_API_KEY

python scripts/ingest_data.py --config configs/dev.yaml
python scripts/build_index.py --config configs/dev.yaml
python scripts/run_api.py --config configs/dev.yaml
```

Open:

http://localhost:8000/docs

---

# 🚀 Setup (Windows – PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

copy .env.example .env
# Edit .env and set OPENAI_API_KEY

python scripts\ingest_data.py --config configs\dev.yaml
python scripts\build_index.py --config configs\dev.yaml
python scripts\run_api.py --config configs\dev.yaml
```

Open:

http://localhost:8000/docs

---

# 📂 Add Documents

Place PDF or TXT files inside:

data/raw/documents/

Then re-run:

```bash
python scripts/ingest_data.py --config configs/dev.yaml
python scripts/build_index.py --config configs/dev.yaml
```

---

# 🔍 Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"How many projects are there?", "top_k": 5}'
```

Example response:

```json
{
  "answer": "There are 3 projects listed.",
  "confidence": 0.92,
  "sources": [
    {
      "chunk_id": "chunk_1",
      "source": "resume.pdf",
      "page": 1,
      "score": 0.88,
      "text": "..."
    }
  ],
  "metrics": {
    "latency_ms": 123,
    "tokens_used": 421,
    "cost_usd": 0.0021
  }
}
```

---

# 🧠 Vector Stores

### Default: Chroma
- Easy local persistence
- Metadata filtering
- Good for local development

### Optional: FAISS
- Faster dense retrieval
- CPU-based

To enable FAISS:

1. Install:

```bash
pip install faiss-cpu
```

2. Update config:

```yaml
vector_store:
  provider: faiss
```

---

# 📊 Evaluation

Place gold dataset in:

evaluation/datasets/gold.jsonl

Run:

```bash
python scripts/run_eval.py --config configs/dev.yaml
```

Generates:

docs/evaluation_results.md

Metrics include:

- Recall@k (dense vs hybrid)
- Precision@k
- Hallucination rate
- Confidence calibration
- P95 latency
- Corpus size

---

# ⚡ Load Testing

1. Start API:

```bash
python scripts/run_api.py --config configs/dev.yaml
```

2. In another terminal:

```bash
python scripts/load_test.py --config configs/dev.yaml
```

Outputs:

- docs/load_test_results.md
- docs/load_test_results.json

Includes:

- P95 latency
- Throughput
- Error rate
- Concurrency handling

---

# 🔐 Environment Configuration

Create `.env` file in project root:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxx
OPENAI_BASE_URL=https://api.openai.com/v1
```

If using a local OpenAI-compatible server (LM Studio / Ollama), adjust `OPENAI_BASE_URL`.

---

# 📁 Key Directories

```text
configs/          → YAML configuration
src/              → Core application code
evaluation/       → Metrics + benchmarking
docs/             → Architecture & tradeoffs
data/             → Raw + processed documents
```

---

# 🧪 Clean Rebuild (if needed)

Delete:

data/processed/indexes/

Then run:

```bash
python scripts/ingest_data.py --config configs/dev.yaml
python scripts/build_index.py --config configs/dev.yaml
```

---

# 📌 Recommended Repository Name

rag-smart-qa

---

# 🏆 Production Features

- Config-driven architecture
- Strict grounding enforcement
- Citation validation
- Hybrid retrieval
- Cost tracking
- Latency monitoring
- Structured logging
- Docker-ready
- Modular and testable design

---

# 📚 Documentation

- docs/architecture.md
- docs/decisions.md
- docs/tradeoffs.md
- docs/security.md
- docs/evaluation_results.md

---

# 📄 License

MIT
