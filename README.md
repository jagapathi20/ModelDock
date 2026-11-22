# ModelDock

An ML inference server built as a backend systems project. The model is the workload; the
subject is the system around it, measured under load.

**Work in progress.** The build is staged; each stage is explained in `docs/stages/`.
This README lists only what is built. The scope of the first release is in
[`docs/DESIGN.md`](docs/DESIGN.md), C1.23.

**Built so far**

- Model runners behind one interface (`load()`, `predict_batch()`): a DistilBERT sentiment
  classifier (the main workload) and a scikit-learn iris classifier (the contrast case).
- A batching benchmark, with every result file committed in `bench/results/` and a test that
  fails if a doc cites a result that isn't there.

**Next, in order:** RPC over Redis queues with timeouts (504), dynamic batching, backpressure
(503), containers and graceful shutdown, Prometheus metrics, open-loop load tests.

**Design:** [`docs/DESIGN.md`](docs/DESIGN.md) holds the initial design and a log of every change
made to it during the build, with the reason and evidence for each.

## Run locally

```bash
uv sync                     # create .venv and install deps
docker compose up -d        # Redis
uv run uvicorn api.main:app --reload
curl localhost:8000/health/live
```

## Models and batching benchmark

```bash
uv sync --extra worker                          # torch (CPU), transformers, scikit-learn
uv run python -m scripts.fetch_sentiment_model  # -> artifacts/sentiment-v1 (pinned Hub commit)
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
