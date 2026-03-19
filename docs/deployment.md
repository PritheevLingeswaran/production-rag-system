# Deployment

## Local Docker Compose

Prerequisites:

- Docker Desktop or Docker Engine with Compose support

Steps:

```bash
cp .env.example .env
docker compose up --build
```

What starts:

- `api` on port `8000`

Operational details:

- The API container runs as a non-root user.
- Uploaded data, configs, and prompts are mounted into the API container for local iteration.

## Running without containers

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make ingest
make index
make api
```

## Render deployment target

Blueprint file:

- [render.yaml](/Users/thamaraiselvang/Pritheev%20Projects/rag-smart-qa/render.yaml)

### API service

- Type: Web Service
- Runtime: Docker
- Root directory: repository root
- Dockerfile: `Dockerfile`
- Blueprint service name: `rag-smart-qa-api`
- Env vars:
  - `RAG_ENV=prod`
  - `RAG_API_KEYS=<comma-separated-api-keys>` to enable API-key protection
  - `API_CORS_ORIGINS=<comma-separated-origins>`
  - `OPENAI_API_KEY` only if hosted generation is desired
  - any provider-specific keys if enabled

Health check path:

- `/readyz`

Persistent disk recommendation:

- Mount a disk for `data/` if you want uploaded files, SQLite state, and indexes to persist across deploys.

## Render deploy steps

1. Push the repository to GitHub.
2. In Render, create a new Blueprint and point it at this repository.
3. Review the generated service from `render.yaml`.
4. Set secret env vars:
   - `RAG_API_KEYS`
   - `API_CORS_ORIGINS`
   - `OPENAI_API_KEY` if using hosted generation
5. Create the persistent disk for `/app/data` on the API service.
6. Deploy the blueprint and wait for `/readyz` to pass.

## Production environment examples

- Root example env: [.env.example](/Users/thamaraiselvang/Pritheev%20Projects/rag-smart-qa/.env.example)

## Deployment caveats

- This repository is deployment-ready for reviewer and small-team use, but not presented as a finished hyperscale platform.
- If you need high-write concurrency, stronger tenancy, or managed object storage, treat the current storage and metadata layers as replaceable seams rather than final infrastructure.
