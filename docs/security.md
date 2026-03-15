# Security And Monitoring

## Security posture in this repo

- Structured error responses avoid leaking raw tracebacks to clients.
- Strict grounding and refusal logic reduce unsupported answer risk.
- Auth remains feature-flagged and abstracted behind `AuthService`.
- Health/readiness checks make startup failures explicit instead of silent.

## Production hardening still recommended

- Use a secrets manager instead of long-lived `.env` files.
- Add real authentication, rate limiting, and tenant-aware document ACL enforcement.
- Add stronger prompt-injection and abuse defenses around hostile uploaded content.
- Replace local storage and SQLite if the deployment requires stronger durability or tenancy guarantees.

## Monitoring

The repository now exposes production-style operational signals:

- Request IDs and correlation IDs on every HTTP response.
- Structured JSON logs with request context.
- Prometheus metrics endpoints at `/metrics` and `/api/v1/metrics`.

Tracked metrics include:

- request count
- HTTP latency
- retrieval latency
- generation latency
- retrieval score diagnostics
- request-stage errors
- refusals
- grounded vs non-grounded answers
- token usage
- cost usage when available

## What is intentionally not claimed

- Full distributed tracing
- Managed SIEM integration
- Secret rotation workflows
- Tenant-isolated security guarantees
