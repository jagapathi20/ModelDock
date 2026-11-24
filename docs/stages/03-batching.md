# Stage 3: Dynamic batching

## Context (read this first in a new chat)

**ModelDock** is an ML inference server built as a **backend systems project**: clients send
inputs over HTTP and get predictions back; the subject is the system around the model (queues,
batching, backpressure, timeouts, failure behaviour), measured under load. The design and every
change to it are in [`docs/DESIGN.md`](../DESIGN.md): Part 1 holds the original decisions
(D1–D14), Part 2 the change log (C0.1 onward, which wins over Part 1), Part 3 current values,
Part 4 open proposals.

Before this stage (Stages 0–2):

- An **API process** (FastAPI) validates each request against a per-version schema from the
  model catalog (`common/models.py`), pushes it onto the Redis list `q:{model}:{version}`, and
  waits up to 500 ms for the reply on its own list `replies:{api_instance_id}`, read by one
  self-supervising listener task (`api/rpc.py`).
- A **worker process** (`python -m worker`) loads one model version and, in Stage 2, popped
  **one request at a time**, checked its deadline, ran the model, and pushed the reply with
  `RPUSH` + `EXPIRE 60` in one pipeline (`worker/loop.py`).
- Two models: DistilBERT sentiment (the workload; batching pays off) and an iris scikit-learn
  classifier (inference ~0.15 ms; batching off).
- Measured so far (all on an Apple M1, model only): a batch of 4 inputs of 48 tokens takes
  38.9 ms at p99, a batch of 8 takes 64.5 ms (run I, C1.14); serving overhead outside inference is
  9.00 ms at p99 at light load on the Compose Redis (C2.6, re-measured in this stage; see
  section 9).

The design decisions this stage implements:

- **D5:** a batch runs when it holds `max_batch_size` requests or `max_wait` has passed since its
  first request, whichever comes first.
- **C1.11:** *one loop, no collector thread.* The worker forms the next batch only once the current
  one is done, so every waiting request stays in Redis, where the API's queue-length check (Stage
  4) can see it. Worst case for a request with d requests ahead: `T × (1 + ceil((d + 1) / B))`.
- **C1.13 / C1.14:** inputs truncated at 48 tokens; `max_batch_size = 4`, `max_wait = 15 ms`.
- **C2.3:** a deadline check at pop and again just before compute; expired requests get no reply.
- **Decided for this stage (C3.1–C3.5):** where the batch window is measured from, and how the
  worker keeps time; thread caps for every native pool; how a failed batch is handled; one
  benchmark script with the current budget; where batching settings live and what each timing
  means.

## What this stage adds

```
modeldock/
├── common/
│   ├── models.py          + max_batch_size, max_wait_ms, max_length per version; runner_options
│   └── protocol.py        + Timings.batch_wait_ms
├── worker/
│   ├── loop.py            rewritten: batch collection, pre-compute deadline check, C3.3 failures
│   ├── threads.py         NEW: cap every native thread pool (P15)
│   ├── __main__.py        thread caps before any runner import; runner built with catalog options
│   └── runners/           build_runner(kind, **options); transformer caps inter-op threads
├── api/main.py            Server-Timing gains batch;dur; X-Batch-Size header
├── scripts/
│   ├── bench_batching.py  merged with bench_batching_1.py (deleted); C1.11 budget
│   └── bench_overhead.py  subtracts batch wait from overhead
└── tests/                 test_worker.py rewritten; test_threads.py new; budget and batching tests
```

## 1. One iteration of the worker loop

```
BLPOP q (≤ 1 s) ──► first message ──► _take: decode · deadline check #1 · schema check
                                      (invalid → its reply now; never reaches the model)
       │
       ▼
LPOP q (B − n)  ──► whatever is already waiting, in one round trip
       │
       ▼ batch not full and window still open?
sleep 1 ms, LPOP q (B − n), repeat       (closes ≤ ~1 ms after it fills, or when the window ends)
       │
       ▼
_compute: deadline check #2 · predict_batch · failure rules (section 4)
       │
       ▼
one pipeline: RPUSH + EXPIRE per reply (including the invalid-input replies)
```

`run_once()` returns how many messages it took off the queue (0 when idle), so tests can drive the
loop one iteration at a time.

## 2. The batch window (C3.1)

### Where it starts

D5 says no request waits more than `max_wait` for its batch to start. "Waits" starts when the
request *arrived*, the API's `enqueued_at_ms`, not when the worker happened to pop it. So time
already spent queued counts against the window:

```python
def batch_window_s(max_wait_ms, enqueued_at_ms, now_wall_ms):
    queued_ms = min(max(now_wall_ms - enqueued_at_ms, 0.0), max_wait_ms)
    return (max_wait_ms - queued_ms) / 1000
```

- **Normal case:** a request queued 0.3 ms (C2.6's queue wait) gets a 14.7 ms window.
- **Behind load:** a request that already queued 40 ms gets **zero**: its batch runs at once with
  whatever is there. Waiting for company would add latency exactly when the system is behind.
- **Clock skew** (`enqueued_at_ms` is the API's wall clock, `now_ms()` the worker's; C2.3):
  the clamp to `[0, max_wait]` bounds the damage. A worker clock *ahead* can at worst close the
  batch at once (as if the request queued long); one *behind* (negative queue time) can at worst
  give the full `max_wait`, never more. Without the clamp, a worker clock 10 s behind would hold a
  batch open for 10 s.

The window is computed once, when the batch's first *valid* request is popped, and then counted
down on the worker's **monotonic** clock (`time.monotonic()`), which skew and wall-clock jumps
can't touch. If the first message is malformed, expired or invalid, the next good one anchors it.

### How the worker waits: polling, not a `BLPOP` timeout

The obvious implementation is `BLPOP q timeout=remaining`: return as soon as a request arrives,
or when the window ends. **It doesn't work, because Redis's blocking timeouts are coarse.** Redis
is single-threaded with an event loop; it checks blocked clients' timeouts from its periodic
timer, which runs `hz` times a second (default 10). A `BLPOP` returns immediately when an element
is pushed, but if nothing arrives its timeout is noticed only at the next timer tick, up to
~100 ms late. On the M1 against the Compose Redis, 1, 5, 15 and 50 ms timeouts all took a median
of about 106 ms, and 200 ms took 210 ms (`bench/results/stage3-redis-blpop-timeout-20261124T084648Z.json`, recorded with `scripts/check_redis_timeouts.py`;
C3.1 has the table). The timeout is
exactly the light-load case, where `max_wait` decides latency, so a 15 ms window would have
become ~100 ms, two thirds of the whole latency budget.

So the worker keeps time itself:

```python
while len(batch) < max_batch_size:
    left = close_at - time.monotonic()
    if left <= 0:
        break
    time.sleep(min(POLL_INTERVAL_S, left))  # 1 ms steps
    take(redis.lpop(queue, max_batch_size - len(batch)))  # non-blocking, returns at once
```

- A batch closes within ~1 ms of filling (one poll step), or exactly at the window's end.
- **Cost:** at light load, up to 15 `LPOP`s per 15 ms window (each a sub-millisecond round trip
  locally). Under load, the first `LPOP` fills the batch and there is no polling at all.
- `BLPOP` is still used for the *first* request of a batch: there an element arriving is what
  matters, and its 1 s timeout only paces the idle heartbeat (`last_loop_at`), where ±100 ms is
  irrelevant.
- Rejected alternatives (C3.1): raising Redis's `hz` (still several ms late, costs Redis CPU, may
  be impossible on a managed Redis); cutting `BLPOP` short with a client socket timeout (redis-py
  treats that as a broken connection and reconnects).

**Interview point:** this is a general lesson about timers in event-loop servers. A timeout
parameter on a server-side blocking call is a *lower* bound on the wait, with the server's timer
resolution deciding the upper bound. When a latency budget depends on a short timeout, keep that
timer in the process that owns the budget.

### Batching off

iris v1 has `max_batch_size = 1`: the `LPOP` and the window are skipped, so it behaves exactly as
in Stage 2 (a test checks it never waits).

## 3. Deadlines and what each timing means (C2.3, C3.5)

- **Check #1, at pop** (in `_take`): an expired request is dropped before it joins a batch.
- **Check #2, before every model call** (in `_compute`): a request can expire *while its batch
  forms*. Dropped requests get no reply; the API answered 504 when its own timer, set from the same
  500 ms, fired.

Each reply carries, for *its* request:

| Field | From → to | Clock | Diagnoses (D13) |
|---|---|---|---|
| `queue_wait_ms` | API enqueue → worker pop | wall (two processes) | overload |
| `batch_wait_ms` | pop → start of the model call that produced the result | worker monotonic | batcher configuration |
| `inference_ms` | that model call | worker `perf_counter` | a slow version |
| `batch_size` | size of that call | | low traffic, batcher config |

The API forwards them in `Server-Timing` (`app`, `queue`, `batch`, `inference`) and adds
`X-Batch-Size`. `scripts/bench_overhead.py` now subtracts `batch` too: with one request in flight
a sentiment request waits out its whole 15 ms window, which is batcher configuration, not path
overhead.

## 4. When a batch fails (C3.3)

The rule: **a request gets an error only if it fails on its own.**

1. **Schema check at pop.** Every popped request is validated against the same catalog schema the
   API uses (`spec.validate_input`), microseconds per input. One that fails gets `invalid_input`
   immediately and never joins the batch. This catches anything that reached the queue without
   going through the API.
2. **The runner rejects one input** (`InvalidInputError` with `index`): that request gets
   `invalid_input`; the others run again *as one batch*. A loop repeats this, so several bad inputs
   are peeled off one at a time, each costing one extra batch.
3. **Anything else fails** (or `InvalidInputError` without a usable index, or the runner returns
   the wrong number of outputs): every request runs **alone**, with a deadline check before each
   run. Each gets its own result; one that fails alone too gets `inference_failed` (500).

`stats.rerun_alone` counts case 3. Each reply's `batch_size` and `inference_ms` describe the model
call that actually produced its result (1 for solo re-runs), and `batch_wait_ms` includes the
time lost to the failed attempt.

## 5. Threads (C3.2, P15)

D7 and C1.5: a worker's threads must fit its share of the performance cores. PyTorch's
`set_num_threads` covers only PyTorch. numpy calls a BLAS library with its own pool (Apple
Accelerate on macOS, OpenBLAS on Linux, sometimes MKL), and OpenMP code (PyTorch on Linux,
scikit-learn) has another. **These read their limits from environment variables once, when the
library is first loaded.** Setting them afterwards does nothing.

`worker/threads.py::cap_native_threads(n)` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` with `os.environ.setdefault`, so an explicit value
in your shell still wins. `worker/__main__.py` calls it before building the runner, and nothing
`worker/__main__.py` imports loads numpy or torch. A test checks exactly that in a fresh
interpreter, because once numpy is in `sys.modules` the caps are too late.

The transformer also gets `torch.set_num_interop_threads(1)`. The inter-op pool runs independent
operators concurrently; one forward pass through a sequential model has none. PyTorch allows
setting it only once, so `configure_cpu_threads` skips it if it's already 1 or can't be changed.

At startup the worker logs the pools it actually loaded, using `threadpoolctl` (a scikit-learn
dependency). It doesn't detect Apple Accelerate, so on macOS the log may not list numpy's pool;
it's a check, not a guarantee.

Run the iris worker with `MODELDOCK_WORKER_THREADS=1`: its inference takes ~0.15 ms, so more threads
only add wake-up cost.

## 6. The catalog (C3.5)

```python
SENTIMENT_V1 = ModelSpec(..., max_chars=1000, max_length=48, max_batch_size=4, max_wait_ms=15.0)
IRIS_V1 = ModelSpec(..., n_features=4, max_batch_size=1, max_wait_ms=0.0)
```

Batching settings are per version (D5), and part of what the version *is*, like its input limits.
`spec.runner_options` gives the runner's constructor arguments, so the worker builds
`build_runner(spec.kind, **spec.runner_options)`: the transformer truncates at 48 tokens.

## 7. The benchmark (C3.4)

`scripts/bench_batching.py` is now the only batching benchmark: Stage 1's two copies are merged
(with the P/E core report) and `bench_batching_1.py` is gone. To reproduce a Stage 1 run exactly,
check out the Stage 1.5 commit, where both scripts and the old budget still exist:

```bash
git log --diff-filter=D --oneline -- scripts/bench_batching_1.py   # the commit that deleted it
git checkout <that commit>^                                        # its parent
```

Its budget is now C1.11's, for `--max-batch-size` (default 4), `--max-wait-ms` (15),
`--overhead-ms` (default 9.00, C2.6's measurement on the Compose Redis, cited next to the
default) and `--workers`:

```
Latency budget (C1.11): max_batch_size=4, max_wait=15 ms, T (full batch p99)=38.9 ms, ...
  light_load                      62.9 ms   ok        max_wait + T + overhead
  saturated_empty_queue           86.8 ms   ok        2T + overhead
  at_queue_limit                 125.7 ms   ok        T × (1 + ceil(D / B)) + overhead
  recommended max_queue_depth: 8 (m=2, workers=1)
```

(These lines are worked by hand from run I's T,
`bench/results/stage1-transformer_classifier-20261110T105742Z.json`, not printed by a new run;
`tests/test_bench.py` checks the function gives exactly C2.6's 8 and 125.7 ms.) If even an empty queue misses the target, it says so
instead of recommending a limit. `max_queue_depth` stays an explicit setting chosen from this
output (C1.17): Stage 4 sets it.

Two defaults keep Stage 1's commands meaning what they meant: `--max-length 128` and `--threads`
= all logical cores. Both are recorded in each result's `args`. For current settings pass
`--max-length 48 --threads 4`.

## 8. Tests

| File | New or changed checks |
|---|---|
| `test_worker.py` | size trigger (4 queued → one batch, no wait); time trigger (1 request → runs after ~60 ms); a late arrival joins; a batch closes soon after it fills; FIFO split 4 + 2; the window formula and both skew directions; deadline at pop and while the batch forms; schema-invalid input answered without the model; one rejected input → the rest re-run as a batch; batch failure → solo re-runs; all fail → 500; malformed first message; batching off never waits |
| `test_threads.py` | `setdefault` keeps explicit values; the worker entry point imports no numpy/torch; numpy's pools respect the cap (fresh interpreter) |
| `test_bench.py` | C1.11 budget with hand-worked numbers; workers scale the limit; no-limit case; C2.6's numbers reproduced |
| `test_models.py` | batching settings and `runner_options`; invalid settings rejected; text models need `max_length` |
| `test_roundtrip_redis.py` | **real Redis + real iris model with batching on**: 24 concurrent requests share model calls (`X-Batch-Size` up to 4, fewer model calls than requests), each gets its own answer |
| `test_runner_transformer.py` | the catalog's options give a runner that truncates at 48 (needs torch) |

Timing tests use generous bounds (a 60 ms window must take 50–150 ms) so a busy laptop doesn't
make them flaky.

## 9. Which Redis a benchmark measured (C3.6)

`localhost:6379` isn't necessarily the Compose container. A Redis installed natively (e.g. with
Homebrew) keeps answering when Docker is stopped, and on macOS both can listen at once. That
happened during this stage. So `bench_overhead.py` and `check_redis_timeouts.py` now record the
server's version, `os` and `hz` in each result (`machine.redis`, from `scripts/redis_info.py`) and
warn when the `os` isn't Linux. Check before measuring: `lsof -nP -iTCP:6379 -sTCP:LISTEN` should
show only Docker.

## 10. How this connects to the next stages

- **Stage 4 (backpressure)** adds the `LLEN` check before the API's `RPUSH`: 503 `overloaded` with
  `Retry-After: 1` once the queue holds `max_queue_depth` requests, set from this stage's
  benchmark output. Because the loop holds no formed batches (C1.11), `LLEN` sees the whole
  backlog.
- **Stage 5** serves `last_loop_at` on a health thread and turns `stop()` into SIGTERM handling,
  and adds the **warm-up batch** before the worker reports ready (D12, C1.21). It's a known gap
  until then: the first forward pass after startup is much slower than the rest (PyTorch allocates
  memory and picks kernels on first use; a manual check showed it clearly, not yet measured), so
  the first requests after a worker starts can miss 150 ms. Stage 5 measures it.
- **Stage 6** turns `WorkerStats` (including `batch_sizes`) and the four timings into Prometheus
  histograms.
- **Stage 7's load test** settles P19 (batch size 4 or 8) and P16 (sklearn batching).

## Commands

```bash
uv sync --extra worker
uv run ruff check . && uv run ruff format --check .
uv run pytest -v
uv run python -m scripts.check_redis_timeouts           # records C3.1's evidence

# Batching, by hand: one sentiment worker, one API
uv run python -m worker                                  # terminal 1 (sentiment v1)
uv run uvicorn api.main:app --port 8000                  # terminal 2
for i in $(seq 8); do
  curl -s -o /dev/null -D - -X POST localhost:8000/v1/models/sentiment/predict \
       -H 'content-type: application/json' -d '{"text": "a great film"}' \
       | grep -i -E 'x-batch-size|server-timing' &
done; wait                                               # several show x-batch-size > 1
```
