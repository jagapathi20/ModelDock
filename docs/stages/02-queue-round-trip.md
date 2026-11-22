# Stage 2: Queue round trip

## Context (read this first in a new chat)

**ModelDock** is an ML inference server built as a **backend systems project**: clients send
inputs over HTTP and get predictions back; the subject is the system around the model (queues,
batching, backpressure, timeouts, failure behaviour), measured under load. The design and every
change to it are in [`docs/DESIGN.md`](../DESIGN.md) (Part 1: original decisions D1–D14; Part 2:
change log C0.1 onward, which wins over Part 1; Part 3: current values; Part 4: open proposals).

Before this stage:

- **Stage 0:** packages `api/` (FastAPI), `worker/`, `common/` (`Settings` from `MODELDOCK_*`
  environment variables); ruff, pytest; Docker Compose.
- **Stage 1:** model runners behind one interface, `load(dir)` and `predict_batch(inputs)`: a
  DistilBERT sentiment classifier and a scikit-learn iris classifier. A batching benchmark.
- **Stage 1.5:** housekeeping. Compose runs only Redis; benchmark results committed and checked.

The design points this stage builds on:

- **D3:** API processes never load models; worker processes do. They talk only through Redis.
- **D4 (RPC over a queue):** each request carries a `request_id` and a `reply_to` naming its API
  process's private reply list, `replies:{api_instance_id}`. The worker pushes the reply there; one
  listener per API process reads it and wakes the waiting handler.
- **C1.16:** one 500 ms timeout sets both the request's `deadline_ms` and the API's 504 timer.
- **C1.18:** the worker sets the reply list's 60 s expiry on every push.
- **C1.19:** the listener supervises itself and drives `/health/ready`; futures are cleaned up in
  `finally`; the listener checks `fut.done()` before resolving.
- **Decided for this stage (C2.1–C2.6):** per-version input validation and error codes, two size
  limits, deadline checks, Redis URL as a secret, the HTTP contract, and how overhead is measured.

**This stage builds the first working request path, without batching**: one request in, through
Redis to a worker, one reply back. Stage 3 adds batching to the same loop.

## What this stage adds

```
modeldock/
├── pyproject.toml           + redis (both processes), + fakeredis (dev), `redis` test marker
├── .env.example             + timeout, body limit, worker settings
├── common/
│   ├── config.py            redis_url is SecretStr; request_timeout_ms, max_body_bytes, worker_*
│   ├── models.py            NEW: the model catalog and per-version input schemas
│   ├── protocol.py          NEW: request/reply messages, key names, error codes, now_ms()
│   └── backoff.py           NEW: capped exponential backoff with full jitter
├── api/
│   ├── main.py              lifespan (Redis clients, listener), predict endpoint, /health/ready
│   ├── rpc.py               NEW: PendingReplies, ReplyListener, RpcClient
│   └── limits.py            NEW: body size limit (raw ASGI middleware)
├── worker/
│   ├── __main__.py          NEW: `python -m worker`
│   ├── loop.py              NEW: the worker loop, drift check
│   └── runners/             InvalidInputError; both runners raise it
├── scripts/bench_overhead.py   NEW: light-load overhead measurement
├── tests/                   7 new test files (below)
└── docs/stages/02-queue-round-trip.md   (this file)
```

## 1. The path of one request

```
client                API process                         Redis                  worker
  │ POST /v1/models/iris/predict                            │                       │
  │──────────────►│ BodySizeLimit (413?)                    │                       │
  │               │ catalog lookup (404?)                   │                       │
  │               │ JSON + schema (422?)                    │                       │
  │               │ register future[request_id]             │                       │
  │               │ RPUSH q:iris:1 {request...} ───────────►│ ◄── BLPOP q:iris:1 ───│
  │               │ await future (≤ 500 ms, else 504)       │                       │ deadline passed? drop
  │               │                                         │                       │ predict_batch([input])
  │               │                                         │◄── RPUSH replies:<id> │
  │               │                                         │    EXPIRE replies:<id> 60
  │               │ listener: BLPOP replies:<id> ◄──────────│                       │
  │               │ future.set_result(reply)                │                       │
  │               │ finally: drop future                    │                       │
  │◄──────────────│ 200 / 422 / 500                         │                       │
```

Two processes, two Redis lists. Nothing in the API blocks: the handler `await`s a future, so
while it waits the event loop serves other requests. That is why one API process can have many
requests in flight, all sharing one reply list, each finding its own reply by `request_id`.

## 2. `common/models.py`: the catalog

There's no registry in the first release (C1.23), so model versions are defined in code:

```python
SENTIMENT_V1 = ModelSpec(
    name="sentiment",
    version=1,
    kind="transformer_classifier",
    artifact="sentiment-v1",
    max_chars=1000,
)
IRIS_V1 = ModelSpec(
    name="iris", version=1, kind="sklearn_classifier", artifact="iris-v1", n_features=4
)
```

- **Why code, not environment variables:** the data is structured (per model, per version), and
  it changes only together with code or artifacts. Environment variables stay for deployment
  knobs. A version is immutable: changing a limit means a new version, because cached answers
  (D10) and benchmark results are tied to "sentiment v1" meaning one thing.
- **`spec.queue`** gives `q:{model}:{version}` (section 6). Keeping the version in the name now
  means canary routing can be added later without renaming queues.
- **`serving_version(name)`** picks the version traffic goes to: with one version per model, that
  one. Canary routing (future work) would replace it.
- **`spec.input_model`** builds a pydantic model from the version's limits with `create_model`,
  once per spec (`cached_property`; it works on a frozen dataclass because it writes to the
  instance `__dict__` directly rather than through `__setattr__`):
  - text: `str`, `strict=True` (no `5` → `"5"`), 1 to `max_chars` characters;
  - features: exactly `n_features` items, each `strict` (JSON integers are accepted, strings and
    booleans aren't) and finite (`allow_inf_nan=False`; Python's `json` module parses `NaN` and
    `Infinity` tokens, so without this a client could send them);
  - `extra="forbid"`: an unexpected field is reported, not ignored.

**Why the API validates at all (C2.1):** a batch fails as a whole if one input is bad. Rejecting
bad input before it's queued means it never shares a batch with anyone. The API can't load the
model to ask (D3), so every limit it checks must be in the catalog. That is why `n_features` is
written twice, here and inside `model.joblib`, and why the worker runs a **drift check** at
startup (`worker/loop.py::check_spec_matches`): if the two disagree, the worker refuses to start.
Otherwise the API would accept every request and the model reject every one.

## 3. `common/protocol.py`: wire formats

Request and reply messages are pydantic models with `encode()` (JSON bytes) and `decode()`.
Decoding validates, so a malformed message fails at the boundary with a clear error.

- **`RequestMessage`**: `request_id` (32 hex), `reply_to`, `model`, `version`, `input`,
  `deadline_ms`, `enqueued_at_ms`.
- **`reply_to` must match `replies:<32 hex>`.** The worker pushes to whatever key a message names.
  Without the pattern, a forged or buggy message could make it push into any key, say
  `q:sentiment:1`, injecting junk into a queue (C2.5).
- **`ReplyMessage`**: `request_id`, `status` (`ok`/`error`), `output`, `error`
  (`{"code", "message"}`), `version`, `timings` (`queue_wait_ms`, `inference_ms`, `batch_size`;
  `batch_wait_ms` arrives with batching).
- **`extra="ignore"`** on messages: a newer process may add a field without breaking an older one
  in the middle of a rolling restart. (Contrast with the client-facing input schema, which forbids
  extras: there, an unknown field is probably a client bug worth reporting.)
- **`now_ms()` is wall-clock time as fractional milliseconds.** Wall clock because `deadline_ms`
  and `enqueued_at_ms` are written by one process and read by another, and a monotonic clock's
  zero point is different in every process. Fractional because whole milliseconds rounded the
  queue wait to 0 or 1 on a path that takes a few milliseconds end to end (C2.5). The cost is the
  **clock-skew limitation** in C2.3: across machines, comparisons are only as good as NTP.
- Constants: `REPLY_TTL_S = 60` (C1.18), `BLOCK_TIMEOUT_S = 1` (every blocking pop wakes at least
  once a second), `HEARTBEAT_STALE_S = 3` (C1.19).

## 4. The worker: `worker/loop.py`, `worker/__main__.py`

`python -m worker [--model iris]` reads settings, finds the catalog entry, sets PyTorch's thread
count (transformer only, before the first inference), loads the runner, runs the drift check,
creates a sync Redis client and runs `Worker.run()`.

**One iteration (`run_once`):**

1. `BLPOP q:{model}:{version}` with a 1 s timeout. `last_loop_at` is updated whether or not a
   request arrived: Stage 5's `/health/ready` uses it as a heartbeat (C1.21).
2. Decode. A malformed message is counted and dropped. It can't be answered: its `reply_to` is
   exactly the part that can't be trusted.
3. **Deadline check (C2.3).** If `now_ms() >= deadline_ms`, drop it: no reply, no compute, count
   it. The API already answered 504, because its timer started from the same clock reading as
   `deadline_ms`. P9 asks for a check at pop *and* just before compute; in Stage 2 those are the
   same moment, and Stage 3's batching wait will separate them.
4. `predict_batch([input])`, timed with `perf_counter`.
5. Reply: `RPUSH replies:<id>` and `EXPIRE replies:<id> 60` in one pipeline (C1.18).
   `transaction=False` sends both in one round trip without `MULTI`/`EXEC`; atomicity isn't
   needed, because if an `EXPIRE` were lost the next push would set it.

**Error mapping (C2.1):**

| Raised by the runner | Reply `error.code` | HTTP |
|---|---|---|
| `InvalidInputError` | `invalid_input`, with the runner's message | 422 |
| anything else (`ValueError` from numpy, `RuntimeError`, `KeyError`...) | `inference_failed`, message `"inference failed"` | 500 |

The traceback for the second row goes to the worker's log, not to the client: internals don't
belong in an HTTP response. Either way **the loop carries on**: one request never kills the worker.

**`InvalidInputError`** (`worker/runners/base.py`) is a `ValueError` subclass, so old code catching
`ValueError` still works, but the worker catches the subclass specifically. Runners raise it only
after checking the input themselves:

- `SklearnClassifierRunner` checks each input (`features` present, a list of the right length,
  numbers, finite) before building the matrix, and reports the bad input's `index`. Previously a
  whole-batch `np.asarray` raised a generic `ValueError` that said neither which input was wrong
  nor whose fault it was.
- `TransformerClassifierRunner.preprocess` checks that each `text` is a non-empty string.

`index` exists for Stage 3: to fail only the request with the bad input.

**Redis errors** (`run()`): back off (capped exponential, full jitter, `common/backoff.py`) and
retry, forever. Exiting would restart the process and reload the model, and wouldn't fix Redis.

**Queue order:** the API appends with `RPUSH`, the worker takes from the head with `BLPOP`, so the
oldest request goes first. Stage 3 takes several at once with `LPOP key count`.

## 5. The API: `api/rpc.py`

### `PendingReplies`: the futures map

`request_id → asyncio.Future`. No lock: the handler and the listener are coroutines on the same
event loop, and between two `await`s nothing else runs.

- `register()` happens **before** the enqueue. Otherwise a fast worker could reply before the
  future exists, and the reply would be thrown away as late.
- `resolve()` checks `fut.done()` before `set_result()`. A handler can time out at the exact
  moment its reply arrives; `set_result` on a cancelled future raises `InvalidStateError`, which
  would kill the listener (C1.19). It returns `False` instead, and the reply counts as late.
- `discard()` uses `pop(id, None)`: safe if the entry is already gone.

### `ReplyListener`: one task per process

```python
while True:
    try:
        item = await redis.blpop([key], timeout=1)
    except Exception:  # ConnectionError, TimeoutError, ...
        await asyncio.sleep(backoff.next_delay())
        continue
    backoff.reset()
    last_iteration = monotonic()
    if item:
        dispatch(item)  # decode errors are caught inside: log, count, skip
```

- `except Exception` catches every error but not `asyncio.CancelledError`, which subclasses
  `BaseException`. So `stop()` (which cancels the task) still works.
- **A dedicated Redis client:** `BLPOP` holds its connection for up to a second. On the shared
  pool it would take a connection away from handlers each time.
- **Socket timeout 2 s > `BLPOP` timeout 1 s**, or every idle wait would surface as a timeout
  error.
- **Heartbeat:** `last_iteration` is updated only when `BLPOP` *returns* (a reply or a normal
  1 s timeout), not when it fails. `healthy()` is true if that was under 3 s ago. So readiness goes
  false when the listener crashed, is stuck, **or** keeps failing because Redis is down. All
  three mean "replies can't reach handlers".
- **Late replies** (the handler already returned 504) find no future and are counted and
  discarded (D4).

### `RpcClient.call()`: what the handler awaits

```python
arrived = now_ms()
request = RequestMessage(..., deadline_ms=arrived + timeout, enqueued_at_ms=arrived)
fut = pending.register(request.request_id)
try:
    async with asyncio.timeout(timeout / 1000):
        await redis.rpush(spec.queue, request.encode())  # failure → EnqueueFailed → 503
        return await fut  # resolved by the listener
except TimeoutError:
    raise RequestTimeout(...)  # → 504
finally:
    pending.discard(request.request_id)  # every exit path cleans up
```

- **One clock reading** sets the deadline the worker checks and starts the handler's timer (C1.16).
  The handler's timer itself runs on asyncio's monotonic loop clock, so clock skew between
  machines (C2.3) can make the *worker* drop a request early or compute it late, but can never make
  the *API* wait longer than 500 ms.
- **The enqueue is inside the timeout:** if Redis hangs rather than refusing connections, the
  request still ends at 500 ms (504) instead of waiting for TCP timeouts.
- **`finally`** runs on success, timeout, failed enqueue, and cancellation (the client
  disconnected and the server cancelled the handler). Cleaning up only on timeout would leak an
  entry for every failed enqueue, fastest during an outage (C1.19).

## 6. The API: `api/main.py` and `api/limits.py`

### Lifespan

FastAPI's `lifespan` runs code at startup and shutdown. At startup it creates two Redis clients
(commands, and the listener's dedicated one), a unique `api_instance_id`, the futures map, the
listener task and the `RpcClient`, all on `app.state`. At shutdown it cancels the listener and
closes the clients. Connections open lazily, so **an API started while Redis is down still comes
up**: live, not ready, and it recovers by itself when Redis returns (tested).

`create_app(settings, redis_factory)`: tests pass a factory that returns an in-memory fake Redis.

### The predict endpoint

1. Body size limit, in middleware: 413 (below).
2. `serving_version(name)`: 404 `unknown_model`.
3. `json.loads(body)` and `spec.validate_input()`: 422 `invalid_input`, with `details` listing each
   error's location and message. Not the `input` pydantic would include: that would echo the
   client's data back.
4. `rpc.call()`: 504 `timeout`, or 503 `backend_unavailable` with `Retry-After: 1` if the enqueue
   failed.
5. Reply: 200 with `{"request_id", "model", "version", "output"}`; `invalid_input` → 422;
   `inference_failed` → 500.

Every error has the same shape, `{"error": {"code", "message", ...}}` (C2.5). Successful responses
carry `X-Request-Id`, `X-Model-Version`, and **`Server-Timing`**:
`app;dur=…, queue;dur=…, inference;dur=…` in milliseconds. `Server-Timing` is a standard header
(browser dev tools display it). `app` is the time inside the handler; the overhead benchmark uses
all three.

**503 for an unreachable queue** is a choice C2.5 records: D6 reserves 503 for overload, but HTTP
defines 503 as "temporarily unable to handle the request", which fits a Redis outage too, and
tells clients to retry. The `code` distinguishes the two causes.

### `BodySizeLimit` (C2.2)

Raw ASGI middleware. An ASGI app gets the request body as a series of `http.request` messages by
calling `receive()`. The middleware:

- checks `Content-Length` first: over the limit → 413 immediately, the body is never read (uvicorn
  enforces Content-Length framing, so a client can't send more than it declared);
- otherwise wraps `receive` to count bytes, which covers chunked bodies with no Content-Length.
  Past the limit it raises internally, and answers 413 if the app hasn't started responding;
- sends `Connection: close` with its 413, so unread body bytes aren't parsed as the next request.

Starlette's `BaseHTTPMiddleware` is not used: it's built around complete requests and responses,
not around watching a body stream.

**Two limits, one job each (C2.2):** body bytes (413, 16 KB) protect memory and JSON parsing,
which happen before any field is checked; `max_chars` (422, 1,000) protects the Redis message and
the tokenizer. 16 KB is above the largest valid sentiment body: 1,000 emoji, each escaped by JSON
as a 12-byte surrogate pair, is 12,012 bytes. So the byte limit never rejects input the schema
would accept (a test checks this).

### `/health/ready`

200 if the listener's heartbeat is fresh (under 3 s), else 503. `/health/live` still checks
nothing (Stage 0's reasoning: restarting can't fix Redis).

## 7. Configuration

New settings (all `MODELDOCK_*`):

| Setting | Default | Used by |
|---|---|---|
| `redis_url` | `redis://localhost:6379/0`, now a `SecretStr` (C2.4) | both |
| `request_timeout_ms` | 500 (C1.16) | API |
| `max_body_bytes` | 16384 (C2.2) | API |
| `worker_model`, `worker_version` | `sentiment`, 1 | worker |
| `worker_threads` | 4 (C1.5) | worker |
| `artifacts_dir` | `artifacts` | worker |

## 8. Measuring overhead: `scripts/bench_overhead.py` (C2.6)

C1.14, C1.16 and C1.17 assume 30 ms of overhead (everything except inference). This script
measures it: one API process, one worker, and **one request at a time**, so nothing ever queues.
For each request it records the client's end-to-end time and the `Server-Timing` values, and
reports p50/p90/p99/max/mean of:

- `e2e_ms`, `inference_ms`, and **`overhead_ms = e2e − inference`**;
- `http_ms = e2e − app`: client, HTTP, uvicorn, middleware;
- `rpc_ms = app − queue − inference`: validation, JSON, Redis round trips, worker decode, listener.

It uses one keep-alive connection (as any pooled client would), skips 50 warm-up requests, and
writes `bench/results/stage2-overhead-<model>-<timestamp>.json` with machine info and provenance.

**It's a lower bound.** Under load, the event loop, Redis and the worker add queueing on top. A
queue limit derived from this number is optimistic: Stage 4 uses it provisionally, and Stage 7's
load test checks guarantee 2 (accepted requests finish within 150 ms under overload; P10).

The sklearn model's inference takes a fraction of a millisecond, so for iris `e2e` is almost
pure overhead, a cross-check for the sentiment figure, which subtracts a much larger inference
time. **No numbers are recorded yet:** C2.6 gets them, with their result files, after the
benchmark runs natively on the M1.

## 9. Tests

| File | What it checks |
|---|---|
| `test_models.py` | catalog lookup, queue names, the schemas: length limits, strict types, NaN/inf, extra fields |
| `test_protocol.py` | message round trips, `reply_to` pattern, unknown error codes, extra fields ignored, fractional `now_ms` |
| `test_body_limit.py` | at the limit passes; declared length over → 413 without running the handler; chunked over → 413; chunked under passes; bad Content-Length → 400 |
| `test_worker.py` | reply + expiry; idle heartbeat; FIFO; expired → no compute, no reply; `InvalidInputError` → `invalid_input`; other exceptions → `inference_failed`; survives failures and Redis errors; malformed messages; drift check |
| `test_reply_listener.py` | resolves futures; skips malformed replies; `done()` race; survives connection errors; heartbeat → health; stop |
| `test_api_predict.py` | full HTTP path with a stand-in worker: 200 with headers, message fields, 504 with no leak, late reply discarded, 422/500 mapping, invalid input never queued, 404, 413, 503 with cleanup, readiness, worst-case body fits |
| `test_roundtrip_redis.py` | **real Redis**: real worker thread + real iris model + API; 30 concurrent requests each get their own answer; 504 leaves the request queued and a later worker drops it; C1.18's Redis rules; API with Redis down (live, not ready, 503) |

**fakeredis** (dev dependency) implements Redis commands in memory, including blocking pops across
threads, so the unit tests need no server. The `redis`-marked tests need a real one at
`TEST_REDIS_URL` (default `redis://localhost:6379/15`, database 15 of the Compose Redis, flushed
before and after each test, so database 0 is untouched) and are skipped if none answers. The
variable has no `MODELDOCK_` prefix because `conftest.py` deletes `MODELDOCK_*` variables for
isolation.

## 10. How this connects to the next stages

- **Stage 3 (batching)** replaces `BLPOP` + one request with: block for the first request,
  `LPOP key count` for more, wait until `max_wait` after the first `enqueued_at_ms` (or decide to
  measure from the worker's own pop time instead; see C2.3), the second deadline check before
  compute, and the C2.1 rule for a failed batch (re-run each request alone). `index` on
  `InvalidInputError` and the per-request reply path are already in place.
- **Stage 4 (backpressure)** adds the `LLEN` check before `RPUSH` in `RpcClient.call`, with its own
  503 code, and recomputes `max_queue_depth` from the overhead measured here.
- **Stage 5** wraps the worker's `last_loop_at` and `stop()` in an HTTP health thread and SIGTERM
  handling.
- **Stage 6** turns `WorkerStats` and `ListenerStats` into Prometheus metrics.

## Commands

```bash
uv lock && uv sync --extra worker          # new deps: redis, fakeredis
docker compose up -d
uv run ruff check . && uv run ruff format --check .
uv run pytest -v                           # includes the real-Redis tests (database 15)

# Run it
uv run python -m scripts.train_iris
uv run python -m worker --model iris       # terminal 1
uv run uvicorn api.main:app --port 8000    # terminal 2
curl -s localhost:8000/health/ready
curl -si -X POST localhost:8000/v1/models/iris/predict \
     -H 'content-type: application/json' -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```
