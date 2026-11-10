# ModelDock: design and design changes

This document has two parts:

- **[Part 1](#part-1-initial-design)** is the **initial design**, as it was fixed before any code
  was written (2026-11-03). It is kept as originally written so the reasoning at the start of the
  project stays visible. The only additions are "Changed later" notes that point to Part 2.
- **[Part 2](#part-2-design-change-log)** records **every change to that design**, stage by stage:
  what it was, what it became, why, and the evidence (a benchmark, a review finding, an external
  fact). Each stage of the build adds its entries here.

[Part 3](#part-3-current-values-at-a-glance) lists the current values of the parameters that have
changed, and [Part 4](#part-4-open-proposals) lists proposals that are still undecided.

Per-stage implementation notes are in [`docs/stages/`](stages/), and benchmark reports in
[`docs/benchmarks/`](benchmarks/).

**Rules for maintaining this document**

1. Part 1 is never edited, except to add a "Changed later" pointer next to an affected decision.
2. A design change gets a Part 2 entry in the same commit as the code that implements it.
3. Every entry names its evidence. Performance figures only come from recorded runs.
4. Proposals not yet agreed go in Part 4, not Part 2.

---

# Part 1: Initial design

Fixed 2026-11-03. Section numbers follow the original design document, whose section 1
covered the working process rather than the design, so this part starts at section 2.

## 2. What we're building

A production-style **ML model inference server**: clients send inputs over HTTP and get predictions back. It's framed as a **backend systems project**. The models are just the workload. The real content is queueing, dynamic batching, backpressure, caching, versioned rollouts, rate limiting, reliability, and observability, all measured under load.

**Models hosted:**
- **Main workload:** a small transformer text classifier (sentiment), e.g. `distilbert-base-uncased-finetuned-sst-2-english` via Hugging Face `transformers` on CPU. Batching pays off here: one input costs X ms, but a batch of 16 costs much less than 16·X.
- **Contrast model:** a tiny scikit-learn classifier on numeric features. Inference is ~1–2 ms, so batching is turned **off** (`max_batch_size = 1`). This shows the registry handles multiple models and gives a case where batching isn't worth it.

> **Changed later:** C1.8 (why the contrast model doesn't batch). See [Part 2](#part-2-design-change-log).

---

## 3. Architecture overview

```
                    ┌────────────────────── Redis ──────────────────────┐
 client             │                                                    │
   │  POST          │  q:sentiment:v1  (list)      replies:api-7f3a (list)│
   ▼                │  q:sentiment:v2  (list)      replies:api-91bc (list)│
┌──────────────┐ enqueue  ───────────────┐   ┌─── push result ───────────┐
│ API process  │──────────►│ queue per  │──►│ Model worker (per version)│
│ (FastAPI)    │           │ model+ver  │   │  collector thread         │
│ - auth       │           └────────────┘   │  compute thread (PyTorch) │
│ - rate limit │◄──── listener on own reply list ◄─────────────────────┘
│ - routing    │
│ - cache      │      Postgres: models, versions, clients, API keys
│ - backpressure│     Object storage (MinIO / S3 / R2): model weights
└──────────────┘      Prometheus + Grafana: metrics
```

**Request path inside the API process, in this exact order:**
1. Authenticate the API key.
2. Rate limit (token bucket per client).
3. Choose the model version (sticky canary routing).
4. Check the response cache. On a hit, return immediately.
5. Check backpressure (queue depth). If over the limit, return 503.
6. Enqueue with `request_id`, `reply_to`, and `deadline`.
7. Await the result future. On timeout, return 504.

---

## 4. Locked design decisions

### D1: Workload and latency target
- The main workload is a CPU transformer where batching gives a large throughput gain (roughly 10× in illustrative numbers) at the cost of somewhat higher per-request latency. That trade-off is the core of the project.
- The tiny sklearn model runs with batching off.
- **Target: p99 < 150 ms at target load** for the main model, from worst case ≈ `max_wait (15 ms) + batch run time + overhead`. Re-check against real measured batch times in Stage 1.

> **Changed later:** C1.1, C1.2, C1.3, C1.9. See [Part 2](#part-2-design-change-log).

### D2: Synchronous API
- `POST /v1/models/{name}/predict` is **synchronous**: the response contains the prediction.
- Total time is ~100 ms, well within normal HTTP. An async job API would add a job store and polling round trips and make latency worse.
- Async fits work taking seconds to minutes, or work that must outlive the client connection. An async bulk-jobs endpoint is **out of scope**.
- Internally, requests still go through a queue for batching. The client never sees that.

### D3: Separate API and worker processes
- **API processes** do HTTP, validation, auth, routing, caching, and backpressure only. They **never load models**.
- **Model worker processes** load models and run batched inference.
- Why: one shared queue sees all traffic, so batches fill. If each API process ran its own model, batches would be fragmented across processes, the model would be duplicated in RAM per process, and 60 ms CPU calls would block FastAPI's event loop. The two layers also scale independently.

> **Changed later:** C1.10 (worker-only dependencies). See [Part 2](#part-2-design-change-log).

### D4: Routing results back (RPC over a queue)
- Each request carries a **`request_id`** (correlation ID) and a **`reply_to`** naming its API process's private Redis reply list, `replies:{api_instance_id}`.
- Each API process generates a unique `api_instance_id` (UUID) at startup.
- Workers **push** each result to that request's `reply_to` list. There's no polling.
- Each API process runs **one background listener** on its reply list and resolves an in-memory map `request_id → asyncio.Future`.
- On timeout, the handler deletes its map entry. **Late results** with no matching entry are discarded.
- Reply lists get an `EXPIRE` that the listener refreshes periodically, so a dead API process's list disappears on its own.
- Redis is the transport for request queues and reply lists.

### D5: Dynamic batching rule
- The worker **blocks until a request arrives**. That first request starts a clock.
- It keeps collecting until **`max_batch_size` is reached or `max_wait` has passed since the first request**, whichever comes first, then runs the batch.
- No request waits more than `max_wait` for its batch to start.
- `max_wait` mostly matters at light load. `max_batch_size` mostly matters at heavy load, where the queue refills while a batch runs.
- Both are configurable **per model version**. Starting values for the main model: `max_batch_size = 16`, `max_wait = 15 ms`. The sklearn model uses `max_batch_size = 1`.

> **Changed later:** C1.4 (`max_batch_size` 16 → 8). See [Part 2](#part-2-design-change-log).

### D6: Backpressure and overload
- Before enqueuing, the API checks queue depth (`LLEN`). Over the threshold, it **immediately returns 503 Service Unavailable with `Retry-After`**, rather than letting requests wait.
- The threshold comes from the latency budget: about worker throughput × allowed queue wait, starting at **16 × number of workers** for that version, configurable. The check-then-push is non-atomic and treated as a soft limit, which is acceptable.
- Every request carries a **`deadline`** (arrival time + client timeout). **Workers drop expired requests** instead of running them.
- 503 means "the server is overloaded." 429 is reserved for per-client rate limiting (D11).
- Documented client behavior: **exponential backoff with jitter** on 503, to avoid retry storms. The load-test client implements it.

> **Changed later:** C1.7 (queue limit derived from the budget, per version). See [Part 2](#part-2-design-change-log).

### D7: CPU usage inside a worker
- Default: **one worker process per model version, one batch at a time**, with PyTorch's internal threads set explicitly via `torch.set_num_threads(N)` so total threads never exceed cores.
- Inside each worker, two threads:
  - A **collector thread** reads Redis and builds batches. It's mostly I/O wait, which releases the GIL.
  - A **compute thread** runs inference.
  - The collector hands full batches to the compute thread through an in-process queue, so collecting the next batch overlaps with computing the current one.
- Why not many Python threads (each running batches): Python pre/post-processing is serialized by the GIL, and multiple threads each spawning PyTorch threads oversubscribe the CPU.
- Why not many single-batch processes by default: each holds a copy of the model in RAM, and traffic splits across them, giving smaller batches.
- Processes × threads-per-process is configurable. The benchmark stage compares **1×8, 2×4, and 4×2** (adjust to the machine's core count).

> **Changed later:** C1.5, C1.6. See [Part 2](#part-2-design-change-log).

### D8: Model registry and storage
- **Weights go in object storage** (MinIO locally, S3-compatible such as Cloudflare R2 in deployment) under **immutable keys** that include version and content hash, e.g. `models/sentiment/v3-<sha256prefix>.pt`. A version's file is never overwritten.
- **Metadata goes in Postgres** (schema in section 5).
- Why not weights in Postgres: it bloats backups and replicas, pushes useful data out of Postgres's memory cache, and ties up a connection during multi-hundred-MB transfers.
- Workers keep a **local disk cache keyed by hash**, downloading only on a miss. Because files are immutable, the cache never needs invalidating, and rollback is fast.
- Workers **verify the SHA-256** after download against the value in Postgres.
- The upload endpoint **stores the file first, then inserts the metadata row**, so a version never points at a missing file.

> **Changed later:** C0.1 (local MinIO image); open proposal P2. See [Part 2](#part-2-design-change-log).

### D9: Versioned rollout (canary)
- **One Redis queue per model version**: `q:{model}:{version}`. Each worker pool serves exactly one version and pulls from its queue.
- Traffic weights (`traffic_percent`) live in Postgres. **API processes cache the routing table in memory** and reload it when a message on the Redis pub/sub channel `routing:changed` arrives. There's no Postgres query on the hot path.
- **Sticky routing:** `hash(client_id) % 100` compared against cumulative weights, so a client consistently hits the same version.
- **Rollback:** set the version's weight to 0 and publish. Its queue drains normally, then its workers can be stopped.
- The backpressure check (D6) applies **per queue**.
- Every response carries **`X-Model-Version`**. All metrics are labeled by version.
- *Reconciliation with D8:* several versions can be `live` at once during a canary. The rule is that the `traffic_percent` values of a model's live versions sum to 100, enforced in a transaction when weights change.

### D10: Response cache
- Redis key: `cache:{model}:{version}:{sha256(canonical_input)}`.
- **No invalidation is ever needed.** A version is immutable, so `(version, input)` always gives the same answer. A new version simply uses new keys.
- The lookup happens **after version routing, before backpressure and enqueue**. Cache hits skip the queue and don't count toward backpressure.
- Inputs are **canonicalized before hashing** (sorted keys, consistent formatting).
- **TTL plus Redis LRU eviction** bound memory only. They're not for correctness.
- Models run in **deterministic eval mode** (no dropout, no sampling).
- Caching is configurable per model (`cache_enabled`).

### D11: Authentication and rate limiting
- **API key format:** `sk_<key_id>_<secret>`. Postgres stores `key_id` and the **SHA-256 hash of the secret**, never the plaintext. The full key is shown **once** at creation.
- Why SHA-256 and not bcrypt: keys are 32 random bytes and can't be brute-forced. Slow hashes exist for low-entropy human passwords, and bcrypt per request would only add latency.
- Lookup is by `key_id`, then a **constant-time comparison** of hashes (`hmac.compare_digest`).
- API processes **cache key → client in memory**. **Revocation** publishes to the Redis channel `apikeys:revoked`, which clears the cached entry immediately.
- **Rate limiting:** a token bucket per client in Redis (capacity = per-minute limit, continuous refill), implemented as **one atomic Lua script** so concurrent API processes can't double-spend the last token. Limits are configurable per client.
- Over the limit returns **429** with `Retry-After` and `X-RateLimit-Remaining`.
- **Fail open:** if Redis is unreachable for the limiter, allow the request and log an error.

### D12: Reliability
- **No recovery of in-flight requests** when a worker crashes. Use a plain pop, not Streams with acks. Detecting the crash and reclaiming requests takes longer than the request deadline, so recovered requests would already be expired, and the ack machinery would add Redis operations to every request for no benefit with a synchronous API. (Revisit only if an async bulk-jobs API is added.)
- When the handler's timeout fires, return **504 Gateway Timeout** immediately.
- Inference is **idempotent** (no side effects), so client retry is safe. Document this.
- Docker `restart: unless-stopped` restarts crashed processes.
- **Graceful shutdown on SIGTERM:**
  - Worker: stop pulling, finish the current batch, push its results, exit.
  - API: stop accepting, wait for in-flight requests (bounded), exit.
- **Health endpoints:**
  - `/health/live`: the process is alive.
  - `/health/ready`: the process can serve. For workers, that means only after the model is downloaded, its hash verified, and **one warm-up batch** run.

### D13: Observability
- **Principle:** break latency into stages, each with its own metric:
  `total = queue wait + batch formation wait + inference + overhead`.
  Overload inflates queue wait, a misconfigured batcher inflates batch formation wait, and a slow version inflates inference.
- **Prometheus metrics, all labeled `model` and `version`:**

| Metric | Type | Diagnoses |
|---|---|---|
| End-to-end request latency | histogram | the symptom (SLO) |
| Queue wait time | histogram | overload |
| Queue depth | gauge | overload |
| Batch formation wait | histogram | batcher config |
| Batch size | histogram | batcher config, low traffic |
| Inference time per batch and per item | histogram | slow version (compare per item, since canary batches are smaller) |
| Requests by outcome: 200, 429, 503, 504, cache_hit | counter | rejections, timeouts, cache rate |

- Workers expose `/metrics` on their own port (`prometheus_client` HTTP server). API processes expose `/metrics` via FastAPI.
- **Structured JSON logs** in both process types, always including `request_id`.
- **Grafana dashboard** with one row per latency stage, plus throughput and outcomes.
- **Alerts:** p99 > 150 ms for 5 minutes, and 503 rate > 1%.

### D14: Deployment (two setups)
**Benchmark setup (local Docker Compose):** the full architecture. Multiple API processes, worker pools per version, Redis, Postgres, MinIO, Prometheus, Grafana, and the Locust load generator. **All performance numbers in the README come from this setup**, with the hardware documented.

**Live demo (free tier, e.g. Render, ~512 MB RAM, sleeps after idle):**
- **Smaller model:** the full transformer plus Python, PyTorch, and two process types won't fit in 512 MB. Use a small distilled model with **dynamic int8 quantization**, and measure its RAM.
- **One container:** the API process and one worker co-located in a single container under a small process supervisor. It's the same code and the same Redis-mediated architecture, just packed into one service.
- **Bake the demo model into the Docker image** as its initial version, so waking from sleep skips the download. The registry upload flow still works.
- **Cold start:** after waking, the service isn't ready until the model loads and warms up. Expose this through `/health/ready`, and have the demo page show a "warming up" state by polling it. Document the expected first-request delay.
- **Managed backing services:** Neon (Postgres), Upstash (Redis, TLS `rediss://`), Cloudflare R2 (objects). **Verify** that the Redis provider supports what we use (Lua `EVAL`, blocking pops, pub/sub), and estimate Redis commands per request against its free quota.
- The demo shows **functionality**, not performance.

> **Changed later:** C0.1 (local MinIO image). See [Part 2](#part-2-design-change-log).

---

## 5. Data model (Postgres)

```
models
  id              PK
  name            text UNIQUE            -- e.g. "sentiment"
  cache_enabled   boolean
  created_at      timestamptz

model_versions
  id              PK
  model_id        FK -> models.id
  version         int                    -- UNIQUE (model_id, version)
  storage_key     text                   -- immutable object key
  sha256          text
  max_batch_size  int
  max_wait_ms     int
  status          enum(staging, live, retired)
  traffic_percent int                    -- live versions of a model sum to 100
  created_at      timestamptz

clients
  id                  PK
  name                text
  rate_limit_per_min  int
  created_at          timestamptz

api_keys
  id              PK
  key_id          text UNIQUE            -- public prefix part
  secret_sha256   text
  client_id       FK -> clients.id
  revoked_at      timestamptz NULL
  created_at      timestamptz
```

Use migrations (Alembic).

> **Changed later:** C1.7 (`model_versions.max_queue_depth`); open proposal P1. See [Part 2](#part-2-design-change-log).

---

## 6. Redis keys and wire formats

| Key / channel | Type | Purpose |
|---|---|---|
| `q:{model}:{version}` | list | request queue per version |
| `replies:{api_instance_id}` | list (with refreshed EXPIRE) | results back to one API process |
| `cache:{model}:{version}:{sha256}` | string (TTL) | response cache |
| `ratelimit:{client_id}` | hash (tokens, last_refill) | token bucket, Lua-managed |
| `routing:changed` | pub/sub channel | reload routing table |
| `apikeys:revoked` | pub/sub channel | evict cached key |

**Request message (JSON):**
```json
{
  "request_id": "uuid",
  "reply_to": "replies:<api_instance_id>",
  "model": "sentiment",
  "version": 2,
  "input": { "text": "..." },
  "deadline_ms": 1762160000123,
  "enqueued_at_ms": 1762160000001
}
```

**Reply message (JSON):**
```json
{
  "request_id": "uuid",
  "status": "ok | error",
  "output": { "label": "POSITIVE", "score": 0.98 },
  "error": null,
  "version": 2,
  "timings": { "queue_wait_ms": 4, "batch_wait_ms": 9, "inference_ms": 52, "batch_size": 14 }
}
```

---

## 7. Tech stack

- Python 3.12, `uv` (or Poetry) for dependencies, `ruff`, `pytest` (+ `pytest-asyncio`).
- FastAPI + Uvicorn for API processes.
- `redis-py`: asyncio client in the API, sync client in worker threads.
- Postgres via SQLAlchemy 2.x (async) + `asyncpg`, with Alembic migrations.
- `boto3` for S3-compatible object storage (MinIO locally).
- PyTorch (CPU) + Hugging Face `transformers` for the main model, scikit-learn for the contrast model.
- `prometheus_client`, `structlog` (JSON logs), Grafana.
- Locust for load testing.
- Docker + Docker Compose.

> **Changed later:** C0.2, C1.10. See [Part 2](#part-2-design-change-log).

---

## 8. Build stages

Each stage ends with: code, tests, `docs/stages/NN-<name>.md`, a commit message, and verify commands.

**Stage 0: Project skeleton.** Repo layout (`api/`, `worker/`, `common/`, `tests/`, `docs/`), dependency setup, linting, config via environment variables (pydantic-settings), Docker Compose with Redis, Postgres, and MinIO. A trivial FastAPI app with `/health/live`.

**Stage 1: Standalone model runner and batching measurement.** Load the main model and the sklearn model behind a common interface (`load()`, `predict_batch(inputs) -> outputs`), in eval mode. A script that measures inference time for batch sizes 1, 2, 4, 8, 16, 32 and prints per-batch and per-item time. *This replaces D1's illustrative numbers with real ones.* Discuss with me whether the p99 target and starting batch settings still make sense.

**Stage 2: Queue round trip without batching.** A synchronous predict endpoint. API: `request_id`, `reply_to`, `deadline_ms`, enqueue, futures map, reply-list listener, timeout returns 504, late results discarded, reply-list EXPIRE refresh. Worker: pop one request, skip if past its deadline, run it, push the reply. Model and version hardcoded from config for now. Tests include a worker that never replies (504 expected) and a late reply (discarded).

**Stage 3: Dynamic batching.** Worker collector thread and compute thread with an in-process handoff queue. The batching rule from D5, per-version config, `torch.set_num_threads`. Reply timings fields filled in. Tests for size-trigger and time-trigger batches.

**Stage 4: Backpressure.** Queue-depth check, 503 with `Retry-After`, configurable threshold. Deadline drops in the worker, counted.

**Stage 5: Registry and storage.** Postgres schema and migrations. Upload endpoint (file to MinIO first, then the metadata row). Worker resolves its version from Postgres, uses the local disk cache keyed by hash, verifies SHA-256, runs a warm-up batch. `/health/ready` reflects that. API resolves models from the registry instead of config.

**Stage 6: Versioned queues and canary routing.** Per-version queues, worker pools per version, `traffic_percent` with the sum-to-100 transaction, in-memory routing table reloaded via `routing:changed`, sticky `hash(client_id) % 100`, `X-Model-Version` header, rollback flow. Admin endpoint to change weights.

**Stage 7: Response cache.** Canonicalize, hash, lookup after routing, TTL, `cache_enabled` per model. Redis configured with `maxmemory` and LRU eviction in Compose.

**Stage 8: Authentication and rate limiting.** `clients` and `api_keys` tables, key creation (shown once), SHA-256 plus constant-time compare, in-memory key cache with `apikeys:revoked` invalidation. Token-bucket Lua script, 429 headers, fail-open path. Request path order exactly as in section 3.

**Stage 9: Reliability.** SIGTERM graceful shutdown for both process types, restart policies, liveness vs. readiness finalized. A test or script that kills a worker mid-load and shows clients get 504s, the worker restarts, and traffic recovers.

**Stage 10: Observability.** All D13 metrics, JSON logs with `request_id` in both processes, Prometheus scrape config, a Grafana dashboard provisioned from files, alert rules.

**Stage 11: Load testing and benchmarks.** Locust scenarios (steady load, ramp to overload, low-traffic trickle), a client implementing exponential backoff with jitter on 503. Measure and record in the README:
- batching on vs. off (throughput and p50/p99)
- `max_wait` and `max_batch_size` sweeps
- process × thread layouts (1×8, 2×4, 4×2)
- cache hit-rate impact
- behavior under overload (rejections vs. latency)
- a canary rollout and rollback under load

**Stage 12: Demo deployment.** The single-container demo image per D14 (quantized small model baked in, process supervisor), managed Postgres, Redis, and object storage, a "warming up" demo page, README section on the architecture, design decisions, benchmark results, and how to run locally.

> **Changed later:** C1.6 and C1.7 add work to Stages 3 and 5; C1.5 changes the Stage 11 layouts. See [Part 2](#part-2-design-change-log).

---

## 9. Out of scope (for now)
Async bulk-jobs API, GPU serving, autoscaling, multi-region, Redis Cluster/sharding, model training pipelines.

---

## 10. Key concepts the design covers
- Why batching trades per-request latency for throughput, and how `max_wait` and `max_batch_size` bound it.
- Why API and model processes are separate (fragmented batches, duplicated memory, blocked event loop).
- RPC over a queue: correlation IDs, reply-to, futures, and late results.
- Load shedding: rejecting fast (503) vs. queueing forever, deadlines, backoff with jitter.
- The GIL, and why one process with PyTorch threads plus a collector/compute split beats many Python threads.
- Immutable artifacts make both the model cache and the response cache need no invalidation.
- Canary routing with sticky hashing and instant rollback.
- Why API keys use SHA-256 but passwords use bcrypt, constant-time comparison, and distributed rate limiting with an atomic Lua token bucket.
- Why in-flight requests aren't recovered on crash (deadline vs. detection time, idempotency).
- Diagnosing latency by stage-level metrics.

---

# Part 2: Design change log

Entry IDs are `C<stage>.<n>`. "Before" quotes or summarizes Part 1.

## Stage 0: project skeleton

### C0.1: Local object storage runs Chainguard's MinIO build (D8, D14)

- **Before:** MinIO for local object storage, implicitly the official `minio/minio` image.
- **After:** `cgr.dev/chainguard/minio`, a free build of the same MinIO server from source.
  It still speaks the S3 API, so D8 is otherwise unchanged. The image has no shell, so there is no
  in-container healthcheck (checked from the host via `/minio/health/live`), and no web console.
- **Why:** MinIO stopped publishing the community-edition Docker image in October 2025, so the
  official image can no longer be pulled.
- **Evidence:** MinIO's repository notice; Chainguard's announcement of free MinIO images.
  Details in [`stages/00-skeleton.md`](stages/00-skeleton.md).

### C0.2: Project name and configuration prefix (section 7)

- **Before:** working title "inference server"; configuration "via environment variables"
  with no naming scheme.
- **After:** the project is **ModelDock**. Every setting is an environment variable named
  `MODELDOCK_<SETTING>` (e.g. `MODELDOCK_REDIS_URL`), read by one pydantic-settings class shared
  by API and worker processes. Credentials are `SecretStr`, so they are masked in logs.
- **Why:** naming; the prefix avoids collisions with platform variables such as `PORT`.

## Stage 1: model runners and batching measurement

All measurements: Apple M1 (4 performance + 4 efficiency cores), model only (no Redis/HTTP).
Full report: [`benchmarks/stage-1-batching.md`](benchmarks/stage-1-batching.md).

### C1.1: Measured batching gain is 4.2×, not ~10× (D1)

- **Before:** "roughly 10× in illustrative numbers."
- **After:** batching 16 requests makes each **4.2×** cheaper (13.6 → 3.2 ms per item; one worker
  goes from 74 to 309 items/s). Gains flatten after 16: going to 32 adds 11% throughput and nearly
  doubles batch time. At the chosen batch size of 8: 3.4×, 253 items/s.
- **Why:** D1 asked for its illustrative numbers to be replaced with measured ones.
- **Evidence:** benchmark runs A, B and C (Table 1 of the report).

### C1.2: Worst-case latency model corrected (D1)

- **Before:** worst case ≈ `max_wait (15 ms) + batch run time + overhead`.
- **After:** that holds only at light load. With T = p99 time of a full batch, and the D7
  collector building the next batch while the current one runs:

  | Situation | Worst case (model side) |
  |---|---|
  | Light load | max_wait + T |
  | Busy (a batch running, the request's batch forming) | 2T |
  | Saturated (a batch running, the next one already formed) | 3T |
  | Each full batch queued ahead in Redis | +T |

- **Why:** under load a request waits behind the batches ahead of it, not only for its own.
- **Note:** a first version of this model (printed by the Stage 1 benchmark script) put
  "saturated" at 2T. It missed the already-formed batch. That was corrected before any decision
  relied on it; the script's budget output is updated in Stage 3.
- **Evidence:** Table 4 of the report.

### C1.3: "Target load" defined (D1)

- **Before:** "p99 < 150 ms at target load", with target load undefined.
- **After:** target load = **at most 70% of a worker's measured throughput**. The 150 ms p99
  target applies there. Saturation is overload, where 503s (D6) are the intended response.
- **Why:** a latency target needs a defined load to be testable (Stage 11).

### C1.4: Main model `max_batch_size` 16 → 8 (D5)

- **Before:** `max_batch_size = 16`, `max_wait = 15 ms`.
- **After:** `max_batch_size = 8`. `max_wait` stays 15 ms until the Stage 11 sweep.
- **Why:** at 16, a saturated worker's worst case is 3T = 164 ms (p99), over 150 ms even with an
  empty queue and before network overhead. At 8 it is 100 ms. Cost: 18% lower peak throughput
  per worker (253 vs 309 items/s).
- **Evidence:** run B, Table 4 of the report.

### C1.5: Inference threads = number of performance cores (D7)

- **Before:** "PyTorch's internal threads set explicitly so total threads never exceed cores";
  benchmark layouts 1×8, 2×4, 4×2.
- **After:** threads per worker × workers = **performance-core count** (4 on the M1). On machines
  where all cores are the same, use the physical core count. Stage 11 layouts on this machine:
  1×4, 2×2, 4×1, plus one 8-thread layout to show the effect under load.
- **Why:** with 8 threads on the M1, each operation waits for the slowest (efficiency) core.
  Compared with 4 threads at batch size 16: median 5% slower, p99 about 2× worse
  (107.8 vs 54.6 ms). With 4 threads, p99 is within 6% of the median.
- **Evidence:** runs B vs C (Table 2), reproduced by run A.

### C1.6: The collector holds at most one formed batch ahead (D7)

- **Before:** "The collector hands full batches to the compute thread through an in-process queue."
- **After:** the collector may hold **at most one formed batch** beyond the one running. It starts
  forming the next batch only once the compute thread has taken the previous one.
- **Why:** batches held inside the worker are invisible to the queue-depth check (D6) and add
  latency. The latency model (C1.2) and the queue limit (C1.7) both assume this bound.
- **Implemented in:** Stage 3.

### C1.7: Queue limit derived from the latency budget, stored per version (D6, section 5)

- **Before:** threshold "about worker throughput × allowed queue wait, starting at
  16 × number of workers".
- **After:** a per-version setting, **`max_queue_depth`** (new column on `model_versions`, added
  in Stage 5; read from config until then). Its default comes from the budget:

  ```
  m = floor((150 ms − overhead) / T) − 2
  max_queue_depth = m × max_batch_size × workers     # 503 when queue length ≥ this
  ```

  Current values (overhead assumed 30 ms until Stage 2 measures it): **8 per worker** for the
  transformer (m = 1), about 426 for the sklearn model on model-only timing.
- **Why:** an intermediate proposal, `max_batch_size × workers`, gave the sklearn model
  (`max_batch_size = 1`) a limit of 1 per worker, which would turn any small burst into 503s for
  the cheapest model. Batch size and queue limit are different quantities. A per-version setting
  was chosen over a fixed floor (a magic number) and over accepting the 503s. TF Serving and Triton
  also configure queue limits per model.
- **Evidence:** "queue limit" section of the report.

### C1.8: Why the contrast model doesn't batch (section 2, D5)

- **Before:** sklearn inference "is ~1–2 ms, so batching is turned off ... a case where batching
  isn't worth it."
- **After:** measured cost is a flat **0.12 ms** per call for 1 to 32 items, so batching *does*
  cut per-item cost (15× at 16). It stays off because waiting up to `max_wait` (15 ms) to fill a
  batch costs far more than the 0.12 ms it would save. `max_batch_size = 1` is unchanged.
- **Evidence:** run S.

### C1.9: Transformer input truncated at 128 tokens (D1, new)

- **Before:** not specified (the model accepts up to 512 tokens).
- **After:** inputs are truncated to `max_length = 128` tokens (configurable).
- **Why:** a batch is padded to its longest input, so one very long input would slow every request
  in its batch. Capping the length bounds the worst-case batch time that C1.2 relies on.

### C1.10: Worker-only dependencies; CPU-only PyTorch (D3, section 7)

- **Before:** "PyTorch (CPU) + Hugging Face transformers" in the tech stack.
- **After:** torch, transformers and scikit-learn are an optional `worker` extra
  (`uv sync --extra worker`), so API installs and images never include them. On Linux, torch comes
  from PyTorch's CPU-only wheel index; PyPI's Linux wheel is the CUDA build with ~3 GB of
  unused NVIDIA libraries.
- **Why:** D3 says API processes never load models; packaging now enforces it.

---

# Part 3: Current values at a glance

| Parameter | Initial design | Current | Change |
|---|---|---|---|
| Main model `max_batch_size` | 16 | **8** | C1.4 |
| Main model `max_wait` | 15 ms | 15 ms | |
| Main model input length | unspecified | **128 tokens** | C1.9 |
| Inference threads (M1) | ≤ cores (8) | **4 (performance cores)** | C1.5 |
| Queue limit | 16 × workers | **`max_queue_depth` per version, from the budget (8 per worker, provisional)** | C1.7 |
| p99 target | < 150 ms at target load | < 150 ms at **≤ 70% of measured throughput** | C1.3 |
| Contrast model `max_batch_size` | 1 | 1 (rationale corrected) | C1.8 |
| Local object storage | MinIO | MinIO (Chainguard build) | C0.1 |

---

# Part 4: Open proposals

Not yet decided. When one is agreed, it moves to Part 2 under the stage that implements it.

- **P1: runner type in the registry (section 5, Stage 5).** The schema has no field saying which
  runner loads a model (transformer vs sklearn). Proposal: a `runtime` column on `models`
  (`transformer_classifier` / `sklearn_classifier`).
- **P2: model artifacts as one archive (D8, Stage 5).** D8's example key
  `models/sentiment/v3-<hash>.pt` assumes one file, but a saved Hugging Face model is a directory
  (`config.json`, `model.safetensors`, tokenizer files). Proposal: store each version as one
  `.tar`, so the immutable key, SHA-256 check and hash-keyed cache work unchanged.
- **P3: per-model input validation in the API (Stages 2–3).** One malformed input in a batch makes
  the whole batch fail. Proposal: the API validates input against a per-model schema before
  enqueueing, and the worker returns a batch failure as an error to every request in it rather
  than leaving them to time out.
