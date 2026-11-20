# ModelDock

A production-style ML inference server built as a backend systems project:
dynamic batching, RPC over Redis queues, backpressure, response caching,
canary rollouts, rate limiting and stage-level observability, measured under load.

Work in progress. The build is staged; each stage is explained in `docs/stages/`.

**Design:** [`docs/DESIGN.md`](docs/DESIGN.md) holds the initial design and a log of every change
made to it during the build, with the reason and evidence for each.

## Run locally

```bash
uv sync                     # create .venv and install deps
docker compose up -d        # Redis, Postgres, MinIO
uv run uvicorn api.main:app --reload
curl localhost:8000/health/live
```

## Models and batching benchmark

```bash
uv sync --extra worker                          # torch (CPU), transformers, scikit-learn
uv run python -m scripts.fetch_sentiment_model  # -> artifacts/sentiment-v1
uv run python -m scripts.train_iris             # -> artifacts/iris-v1
uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1
```

## Benchmark results

- [Stage 1: batching and CPU threads](docs/benchmarks/stage-1-batching.md): on an Apple M1,
  batching 16 requests of ~26 tokens makes each 4.2× cheaper, and limiting PyTorch to the
  4 performance cores cuts p99 batch time at batch size 16 from 107.8 ms to 54.6 ms. The gain
  depends on input length: 10.6× at batch 32 for 10-token inputs, 1.8× for 128-token inputs,
  because batch time tracks total tokens (see [`docs/DESIGN.md`](docs/DESIGN.md), C1.13).

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
