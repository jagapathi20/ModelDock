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

- A synchronous predict endpoint over Redis queues: per-model input validation (422), a body
  size limit (413), a 500 ms timeout (504), and a reply listener that recovers from Redis
  outages by itself.

- Dynamic batching in the worker: up to `max_batch_size` requests per model call, a `max_wait`
  window timed by the worker itself, and failed batches re-run so one bad input fails only its
  own request.

**Next, in order:** backpressure
(503), containers and graceful shutdown, Prometheus metrics, open-loop load tests.

**Design:** [`docs/DESIGN.md`](docs/DESIGN.md) holds the initial design and a log of every change
made to it during the build, with the reason and evidence for each.

## Run locally

```bash
uv sync --extra worker      # create .venv; the worker extra adds torch, transformers, sklearn
docker compose up -d        # Redis
uv run python -m scripts.train_iris                  # -> artifacts/iris-v1 (milliseconds)
uv run python -m worker --model iris                 # terminal 1: one worker for iris v1
uv run uvicorn api.main:app --port 8000              # terminal 2: one API process
curl localhost:8000/health/ready                     # {"status":"ready"}
curl -X POST localhost:8000/v1/models/iris/predict \
     -H 'content-type: application/json' -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

For the sentiment model, fetch it first (below), then run `uv run python -m worker` (the default
model) and send `{"text": "a great film"}` to `/v1/models/sentiment/predict`.

## Models and batching benchmark

```bash
uv sync --extra worker                          # torch (CPU), transformers, scikit-learn
uv run python -m scripts.fetch_sentiment_model  # -> artifacts/sentiment-v1 (pinned Hub commit)
uv run python -m scripts.train_iris             # -> artifacts/iris-v1
uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1 \
    --threads 4 --max-length 48 --words 46     # current settings (C1.5, C1.13)
```

## Benchmark results

- [Stage 1: batching and CPU threads](docs/benchmarks/stage-1-batching.md): on an Apple M1,
  batching 16 requests of ~26 tokens makes each 4.2× cheaper, and limiting PyTorch to the
  4 performance cores cuts p99 batch time at batch size 16 from 107.8 ms to 54.6 ms. The gain
  depends on input length: 10.6× at batch 32 for 10-token inputs, 1.8× for 128-token inputs,
  because batch time tracks total tokens (see [`docs/DESIGN.md`](docs/DESIGN.md), C1.13).

- [Stage 2: serving overhead](docs/DESIGN.md#c26-overhead-is-measured-at-light-load-and-treated-as-a-lower-bound-c114-c116-c117-p10):
  on an Apple M1 at light load (one request in flight), the full path (HTTP, API, Redis, worker,
  reply) adds 1.77 ms at p99 on top of inference for the sentiment model, and 1.10 ms for iris.
  A lower bound: overhead grows under load. Results: `bench/results/stage2-overhead-sentiment-20261122T093415Z.json`, `bench/results/stage2-overhead-iris-20261122T092735Z.json`.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
