# Stage 1: Model runners and batching measurement

## Context (read this first in a new chat)

**ModelDock** is an ML model inference server built as a **backend systems project**.
Clients send inputs over HTTP and get predictions back. The interesting parts are the systems
around the model: a Redis queue between API and worker processes, dynamic batching,
backpressure, caching, canary rollouts, rate limiting and observability, all measured under load.

Relevant design decisions for this stage (from the design handoff):

- **D1:** the main workload is a CPU transformer sentiment classifier
  (`distilbert-base-uncased-finetuned-sst-2-english`), where batching raises throughput a lot
  at the cost of some per-request latency. Target: **p99 < 150 ms** at target load. A tiny
  scikit-learn classifier is the contrast case: inference is so cheap that batching is off.
- **D3:** API processes never load models. Only **worker processes** do.
- **D5:** a worker collects requests into a batch until `max_batch_size` (start: 16) is reached
  or `max_wait` (start: 15 ms) has passed since the first request, then runs the batch.
- **D7:** one worker process per model version, PyTorch threads set with
  `torch.set_num_threads(N)` so processes × threads never exceeds the core count.
- **D10:** the response cache relies on inference being deterministic.

Stage 0 built the skeleton: `api/` (FastAPI, `/health/live`), `worker/` (empty), `common/`
(`Settings` from `MODELDOCK_*` env vars), Docker Compose with Redis, Postgres and MinIO.

**Stage 1 adds no server code.** It builds the piece the worker will drive, a *runner* that loads
a model and runs one batch, and a benchmark script that measures how batch size affects
inference time on *your* machine. Those numbers replace D1's illustrative "roughly 10×" and
decide whether the 150 ms target and the starting batch settings hold up.

## What this stage adds

```
modeldock/
├── pyproject.toml                  + [worker] extra (torch, transformers, sklearn), CPU torch index
├── worker/runners/
│   ├── __init__.py                 build_runner(kind) factory, lazy imports
│   ├── base.py                     ModelRunner interface: load(), predict_batch()
│   ├── transformer.py              TransformerClassifierRunner + configure_cpu_threads()
│   └── tabular.py                  SklearnClassifierRunner
├── scripts/
│   ├── fetch_sentiment_model.py    Hugging Face Hub -> artifacts/sentiment-v1/
│   ├── train_iris.py               trains the contrast model -> artifacts/iris-v1/
│   └── bench_batching.py           the measurement script
├── tests/
│   ├── test_runner_transformer.py  uses a tiny random DistilBERT built in the test
│   ├── test_runner_tabular.py
│   └── test_bench.py
└── docs/stages/01-model-runner.md  (this file)
```

`artifacts/` (model files) and `bench/results/` (benchmark JSON) are git-ignored.

## 1. Dependencies: a `worker` extra, and CPU-only PyTorch

```toml
[project.optional-dependencies]
worker = ["torch>=2.15", "transformers>=5.0", "scikit-learn>=1.9", "joblib>=1.6", "numpy>=2.0"]
```

These are an **optional extra**, not regular dependencies. `uv sync` installs only what the API
needs; `uv sync --extra worker` adds the ML libraries. This is D3 showing up in packaging: the
API process never loads a model, so its install (and its Docker image in Stage 9) doesn't carry
PyTorch at all. The `worker` package is still importable without the extra. Only the runner
modules need torch, and `worker/runners/__init__.py` imports them lazily.

**CPU-only torch.** On Linux, `pip install torch` from PyPI gives the CUDA build, which pulls in
about 3 GB of NVIDIA libraries. We serve on CPU, so that's dead weight, and painful in Docker.
PyTorch publishes CPU-only wheels on its own index, and uv can route one package to it:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

- `explicit = true` means the index is used **only** for packages that name it in
  `[tool.uv.sources]`. Without it, uv would also look there for every other package.
- The `marker` limits this to Linux. On macOS and Windows, PyPI's torch wheels are already
  CPU-only, so PyPI is fine. `uv.lock` records the right wheel for each platform.

## 2. The runner interface: `worker/runners/base.py`

```python
class ModelRunner(ABC):
    kind: str

    def load(self, model_dir: Path) -> None: ...
    def predict_batch(self, inputs: Sequence[Mapping[str, Any]]) -> list[dict]: ...
```

Every kind of model sits behind these two calls, so the worker's batching loop (Stage 3) can
drive a transformer or a sklearn pipeline without knowing which. An ABC (abstract base class) is
Python's closest thing to a Go interface with some shared code attached: subclasses must
implement the `@abstractmethod`s, or instantiating them raises `TypeError`.

The contract, written in the class docstring:

1. **`load()` reads from a local directory only. It never downloads.** Getting weights onto disk
   is a separate step: `scripts/fetch_sentiment_model.py` now, the object-storage download with
   SHA-256 check in Stage 5. Keeping them apart means the runner code is the same whatever the
   storage backend, and a worker can't silently hit the network at load time.
2. **`predict_batch()` returns exactly one output per input, in the same order.** The worker will
   match outputs back to requests (and their `request_id`s) by position, so order is part of the
   contract, and there are tests for it.
3. **Inputs and outputs are plain JSON dicts**, the same shape as the `"input"` and `"output"`
   fields of the Redis wire messages. No conversion layer between the queue and the runner.
4. **Deterministic inference** (eval mode, no sampling), because the response cache depends on it.

`predict_batch` before `load` raises `ModelNotLoadedError` rather than an obscure
`'NoneType' object is not callable`.

`build_runner(kind)` in `__init__.py` creates a runner from a string. The imports are inside the
function so `import worker.runners` doesn't load torch. Loading torch takes a couple of seconds
and a few hundred MB of RAM, which you don't want in code paths that just need the interface.

## 3. The transformer runner: `worker/runners/transformer.py`

One batch goes through three phases, as three methods:

| Phase | What happens | Cost profile |
|---|---|---|
| `preprocess` | tokenizer turns texts into padded integer tensors | CPU; Rust code inside the tokenizer library |
| `forward` | the neural network computes logits | CPU-heavy matrix multiplies, multithreaded by PyTorch |
| `postprocess` | softmax, pick the top label | tiny |

`predict_batch` just chains them. They're separate methods so the benchmark can time each phase.

**Why batching makes `forward` cheaper per item.** The model is mostly matrix multiplies. With
batch size 1, each layer multiplies a `[1 × seq_len × 768]` activation by its weight matrices; with
batch size 16, a `[16 × seq_len × 768]` one. The weights (~66M parameters for DistilBERT) have to be
read from memory either way, and for small batches that memory traffic and per-operation
overhead dominate. A bigger batch reuses each loaded weight for 16 rows and keeps the CPU's
vector units busy. So the time for 16 items is much less than 16× the time for 1. How much less
depends on your CPU, which is what the benchmark measures.

**Details worth knowing:**

- **`model.eval()`** switches off dropout (which randomly zeroes activations during training).
  Without it, the same text gives a different score on every call and the response cache would be
  wrong. A test checks `runner._model.training is False`.
- **`torch.inference_mode()`** stops PyTorch from recording the operations for backpropagation.
  It's a stricter, faster form of `no_grad()`. It saves memory and some time.
- **`padding=True`** pads every text in a batch to the length of the **longest** one, and returns an
  `attention_mask` (1 = real token, 0 = padding) that tells the model to ignore the padding. Two
  consequences:
  - Correctness: a text's result must not depend on what else is in its batch. The test
    `test_batched_result_matches_one_at_a_time` checks exactly this.
  - Cost: one long text makes the whole batch pay for its length. That's why
    **`max_length = 128`** (DistilBERT allows 512): it caps the worst-case cost of any batch.
    Longer texts are truncated, keeping the start. This is a new setting that isn't in the
    locked design; it's configurable, and Stage 3 moves it into config.
- **Labels** come from `model.config.id2label` (`{0: "NEGATIVE", 1: "POSITIVE"}` for SST-2), so
  the runner works for any sequence-classification model, not just this one.
- **`local_files_only=True`** in `from_pretrained` makes loading fail fast if files are missing,
  instead of quietly contacting the Hugging Face Hub.

**`configure_cpu_threads(n)`** is a module-level function, not a runner argument, because
`torch.set_num_threads` is **process-wide**. Two runners in one process can't have different
thread counts, so pretending they could (via a constructor argument) would be misleading. It
also sets `TOKENIZERS_PARALLELISM=false`: fast tokenizers run their own Rust thread pool for batch
encoding, which would add threads on top of PyTorch's and break D7's
"threads ≤ cores" budget. For batches of short texts, that pool buys almost nothing.

> **Interview note: a subtle point about D10 (the cache).** Batched and single-item runs give the
> same *label*, but the score can differ in the 6th–7th decimal place. A different batch shape can
> take a different, equally valid path through the CPU's matrix-multiply kernels, and floating-point
> addition isn't associative. That's why the test compares scores with `abs=1e-5`, not `==`. For the
> cache this is harmless: the first result computed for an input is stored and served from then on,
> so clients still see one consistent answer per `(version, input)`.

## 4. The sklearn runner: `worker/runners/tabular.py`

Input `{"features": [5.1, 3.5, 1.4, 0.2]}`, output `{"label": "setosa", "score": 0.97}`.

The artifact is one file, `model.joblib`, holding a dict: the fitted pipeline (`StandardScaler` →
`LogisticRegression`), the label names, and the expected feature count. The scaler is *inside* the
pipeline, so its fitted mean and standard deviation travel with the model and clients send raw
features. A model whose preprocessing lives in separate code is a classic production bug: the
two drift apart.

The runner checks the feature count and raises a clear `ValueError` on mismatch. (Stage 2's API
will validate inputs before they're queued; see "open questions" for why that matters.)

**Security note: `joblib.load` is `pickle` underneath, and unpickling can execute arbitrary
code.** A malicious `.joblib` file is a remote-code-execution vector. Rule: only load artifacts you
produced. Stage 5 adds the second guard: the file's SHA-256 must match the registry. (Transformer
weights are `.safetensors`, a format designed to be safe to load, which is one reason the HF
ecosystem moved to it.)

`scripts/train_iris.py` trains it on the iris dataset, which ships inside scikit-learn, so no
download. Iris is a toy, and that's the point. This model exists to show the interface handles
more than one kind of model, and to give a case where inference costs well under a millisecond,
so batching has nothing to save.

## 5. The benchmark: `scripts/bench_batching.py`

```bash
uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1
```

What it does, per batch size (default 1, 2, 4, 8, 16, 32):

1. Runs `--warmup` batches (default 5) that aren't recorded. The first calls are slower: memory
   gets allocated, caches are cold, and PyTorch picks kernels. Measuring them would mix one-time
   costs into steady-state numbers. (D12's warm-up batch before `/health/ready` exists for the same
   reason.)
2. Runs `--iters` timed batches (default 100), each with **fresh inputs**, timed with
   `time.perf_counter_ns()`, a monotonic high-resolution clock (never use `time.time()` for
   durations; it can jump when the system clock is adjusted).
3. Reports:
   - **p50 / p90 / p99 batch time.** Percentiles, not just the mean, because the SLO is on p99 and
     the mean hides tail spikes. With 100 iterations, p99 is roughly the second-worst sample, so
     treat it as indicative. The script warns when `--iters < 100`.
   - **ms/item** = p50 batch time ÷ batch size, and **items/s** = throughput of a worker that runs
     back-to-back batches of that size.
   - **speedup** = per-item time at batch size 1 ÷ per-item time at this size. This is the real
     version of D1's "roughly 10×".
   - For the transformer: median time per phase (tokenize / forward / postprocess) and the mean
     padded sequence length.
4. Writes everything, **with hardware info** (CPU model, core count, thread setting, library
   versions), to `bench/results/stage1-<kind>-<timestamp>.json`. Any number that goes in the README
   must trace back to one of these files.

Inputs are random sentences of `--words` words (default 24, roughly a short review) from a fixed
vocabulary with fixed seeds, so runs are repeatable. The model's cost depends on token count,
not meaning, so nonsense sentences are fine.

### The latency budget check, and a correction to D1's formula

The script ends by plugging the measured p99 time of a full batch, **T** (at `--max-batch-size`,
default 16), into worst-case formulas:

| Situation | Worst case (model side) | Why |
|---|---|---|
| Light load | `max_wait + T` | The request starts a batch, waits up to `max_wait` for it to fill, then the batch runs. |
| Saturated | `2T` | The compute thread is busy with a full batch. The request's batch fills instantly but waits for that one to finish, then runs. |
| At the backpressure threshold | `3T` | D6 allows `16 × workers` requests queued, which is one more full batch per worker ahead of you. |

**D1's formula, `max_wait + batch time + overhead`, is only the light-load row.** Under heavy load,
`max_wait` stops mattering (batches fill instantly), but a request waits behind the batch that's
already running. With the collector/compute overlap from D7, that's up to one extra batch. With a
full queue at the D6 threshold, it's one more. So the real bound at load is about `2T` to `3T`
plus overhead.

This isn't a design change by itself. It changes **which number to compare against 150 ms**:
`3T + overhead`, not `15 + T`. The table above is the formula the script prints, with your
measured T filled in. Overhead (Redis round trips, JSON, HTTP) isn't measured here; Stage 11 measures
the full path.

## 6. Tests

- **`test_runner_transformer.py`** builds a **tiny random DistilBERT** in a session fixture: a
  19-token vocabulary, 2 layers, 32-dimensional, saved with `save_pretrained` to a temp dir. It
  downloads nothing and runs in about a second. Its predictions are meaningless; the tests check
  the runner's behavior:
  output shape and keys, empty batch, error before `load`, eval mode, determinism, batched ==
  one-at-a-time (the padding and attention-mask check), truncation to `max_length`, padding to
  the longest input (with an exact attention-mask check), `build_runner`, `configure_cpu_threads`.
- **`test_runner_tabular.py`** trains the real iris model into a temp dir (it takes milliseconds)
  and checks known examples, output order, batched == single, feature-count validation.
- **`test_bench.py`** checks the benchmark's logic, not its numbers: the percentile function,
  repeatable input generation, an end-to-end `measure` + `render` run on the iris model, and the
  three budget formulas with hand-picked inputs.
- Each file starts with `pytest.importorskip("torch")` (or `"sklearn"`), so `uv run pytest`
  without the worker extra **skips** these tests instead of failing on import.

The real DistilBERT isn't used in tests: a 260 MB download is wrong for a unit test suite. The
benchmark run is the real-model check.

## Open questions for later stages (not changed in this stage)

1. **The registry needs to know which runner to use.** The locked schema (`models`,
   `model_versions`) has no field saying "this is a transformer" vs "this is sklearn".
   Proposal for Stage 5: add `runtime text` to `models`, holding the runner kind
   (`transformer_classifier` / `sklearn_classifier`).
2. **A transformer artifact is a directory, not one file.** D8's example key
   `models/sentiment/v3-<hash>.pt` assumes a single file, but a saved HF model is several files
   (`config.json`, `model.safetensors`, tokenizer files). Proposal for Stage 5: upload each version as
   one `.tar` archive, so the immutable key, the SHA-256 and the local cache keyed by hash all still
   work unchanged.
3. **One bad input shouldn't fail a whole batch.** If one input in a batch of 16 is malformed, the
   runner raises and all 16 fail. Stage 2's API validates inputs (per-model pydantic schema)
   before enqueueing, so bad input never reaches a batch. Stage 3 also catches batch failures so
   the error goes back to every request in that batch instead of hanging them until 504.

## Commands

```bash
uv sync --extra worker                                   # adds torch (CPU), transformers, sklearn
uv run ruff check . && uv run ruff format --check .
uv run pytest -v

uv run python -m scripts.fetch_sentiment_model           # -> artifacts/sentiment-v1 (~270 MB)
uv run python -m scripts.train_iris                      # -> artifacts/iris-v1

uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1
uv run python -m scripts.bench_batching --kind sklearn_classifier --model-dir artifacts/iris-v1
```

Experiments worth running, since each one answers a design question:

```bash
# Thread layouts for D7 (use your core count; nproc on Linux/macOS)
uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1 --threads 1
uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1 --threads 4

# Input length: how much padding/truncation matters
uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1 --words 8
uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1 --words 96
```

Close other heavy programs while benchmarking, keep a laptop plugged in (battery mode throttles
the CPU), and run each configuration twice. If the two runs disagree a lot, the machine is noisy.

## Results

Measured results, the follow-up thread experiment, and the design changes they led to are in
[`docs/benchmarks/stage-1-batching.md`](../benchmarks/stage-1-batching.md).

Changes made after the first run on macOS:

- `tests/test_runner_tabular.py`: the batched-vs-single test compared sklearn scores with exact
  `==` and failed on macOS (scores differed in the 16th digit, from a different BLAS code path).
  It now checks labels exactly and scores within `1e-9`.
- `scripts/bench_batching_1.py`: a copy of the benchmark that also reports the Apple Silicon
  chip and its performance/efficiency core split. Measurement logic is unchanged.
