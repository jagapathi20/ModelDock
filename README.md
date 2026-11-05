# ModelDock

A production-style ML inference server built as a backend systems project:
dynamic batching, RPC over Redis queues, backpressure, response caching,
canary rollouts, rate limiting and stage-level observability, measured under load.

Work in progress. The build is staged; each stage is explained in `docs/stages/`.

## Run locally

```bash
uv sync                     # create .venv and install deps
docker compose up -d        # Redis, Postgres, MinIO
uv run uvicorn api.main:app --reload
curl localhost:8000/health/live
```

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
