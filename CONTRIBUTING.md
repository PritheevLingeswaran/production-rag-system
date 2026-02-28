# Contributing

## Setup
```bash
make install
cp .env.example .env
```

## Quality gates
- `make lint` (ruff + mypy)
- `make test` (pytest)

## Expectations
- Keep behavior config-driven.
- Add tests for new logic.
- Prefer simple, explicit code over framework magic.
