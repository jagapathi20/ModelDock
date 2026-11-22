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

> **Changed later:** C1.11 (no collector thread; the worker runs one loop), C2.1 (input validated
> against the catalog before enqueue), C2.2 (body size limit before everything). See [Part 2](#part-2-design-change-log).

---

## 4. Locked design decisions

### D1: Workload and latency target
- The main workload is a CPU transformer where batching gives a large throughput gain (roughly 10× in illustrative numbers) at the cost of somewhat higher per-request latency. That trade-off is the core of the project.
- The tiny sklearn model runs with batching off.
- **Target: p99 < 150 ms at target load** for the main model, from worst case ≈ `max_wait (15 ms) + batch run time + overhead`. Re-check against real measured batch times in Stage 1.

> **Changed later:** C1.1, C1.2, C1.3, C1.9, C1.11, C1.12, C1.13. See [Part 2](#part-2-design-change-log).

### D2: Synchronous API
- `POST /v1/models/{name}/predict` is **synchronous**: the response contains the prediction.
- Total time is ~100 ms, well within normal HTTP. An async job API would add a job store and polling round trips and make latency worse.
- Async fits work taking seconds to minutes, or work that must outlive the client connection. An async bulk-jobs endpoint is **out of scope**.
- Internally, requests still go through a queue for batching. The client never sees that.

> **Changed later:** C2.5 (the HTTP contract: request and response bodies, error shape, status
> codes, headers). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C1.18 (the worker sets the expiry on each push), C1.19 (the listener supervises itself). See [Part 2](#part-2-design-change-log).

### D5: Dynamic batching rule
- The worker **blocks until a request arrives**. That first request starts a clock.
- It keeps collecting until **`max_batch_size` is reached or `max_wait` has passed since the first request**, whichever comes first, then runs the batch.
- No request waits more than `max_wait` for its batch to start.
- `max_wait` mostly matters at light load. `max_batch_size` mostly matters at heavy load, where the queue refills while a batch runs.
- Both are configurable **per model version**. Starting values for the main model: `max_batch_size = 16`, `max_wait = 15 ms`. The sklearn model uses `max_batch_size = 1`.

> **Changed later:** C1.4 (`max_batch_size` 16 → 8), C1.11 (what `max_wait` bounds), C1.14 (`max_batch_size` 8 → 4). See [Part 2](#part-2-design-change-log).

### D6: Backpressure and overload
- Before enqueuing, the API checks queue depth (`LLEN`). Over the threshold, it **immediately returns 503 Service Unavailable with `Retry-After`**, rather than letting requests wait.
- The threshold comes from the latency budget: about worker throughput × allowed queue wait, starting at **16 × number of workers** for that version, configurable. The check-then-push is non-atomic and treated as a soft limit, which is acceptable.
- Every request carries a **`deadline`** (arrival time + client timeout). **Workers drop expired requests** instead of running them.
- 503 means "the server is overloaded." 429 is reserved for per-client rate limiting (D11).
- Documented client behavior: **exponential backoff with jitter** on 503, to avoid retry storms. The load-test client implements it.

> **Changed later:** C1.7 (queue limit derived from the budget, per version), C1.11 (formula), C1.14 (value), C1.16 (timeout value), C1.17 (explicit setting, no computed default), C2.3 (deadline checks; wall-clock timestamps), C2.5 (503 also when the queue is unreachable). See [Part 2](#part-2-design-change-log).

### D7: CPU usage inside a worker
- Default: **one worker process per model version, one batch at a time**, with PyTorch's internal threads set explicitly via `torch.set_num_threads(N)` so total threads never exceed cores.
- Inside each worker, two threads:
  - A **collector thread** reads Redis and builds batches. It's mostly I/O wait, which releases the GIL.
  - A **compute thread** runs inference.
  - The collector hands full batches to the compute thread through an in-process queue, so collecting the next batch overlaps with computing the current one.
- Why not many Python threads (each running batches): Python pre/post-processing is serialized by the GIL, and multiple threads each spawning PyTorch threads oversubscribe the CPU.
- Why not many single-batch processes by default: each holds a copy of the model in RAM, and traffic splits across them, giving smaller batches.
- Processes × threads-per-process is configurable. The benchmark stage compares **1×8, 2×4, and 4×2** (adjust to the machine's core count).

> **Changed later:** C1.5, C1.6, C1.11 (collector thread removed). See [Part 2](#part-2-design-change-log).

### D8: Model registry and storage
- **Weights go in object storage** (MinIO locally, S3-compatible such as Cloudflare R2 in deployment) under **immutable keys** that include version and content hash, e.g. `models/sentiment/v3-<sha256prefix>.pt`. A version's file is never overwritten.
- **Metadata goes in Postgres** (schema in section 5).
- Why not weights in Postgres: it bloats backups and replicas, pushes useful data out of Postgres's memory cache, and ties up a connection during multi-hundred-MB transfers.
- Workers keep a **local disk cache keyed by hash**, downloading only on a miss. Because files are immutable, the cache never needs invalidating, and rollback is fast.
- Workers **verify the SHA-256** after download against the value in Postgres.
- The upload endpoint **stores the file first, then inserts the metadata row**, so a version never points at a missing file.

> **Changed later:** C0.1 (local MinIO image); open proposal P2; C1.23 (deferred to future
> work; MinIO removed from Compose). See [Part 2](#part-2-design-change-log).

### D9: Versioned rollout (canary)
- **One Redis queue per model version**: `q:{model}:{version}`. Each worker pool serves exactly one version and pulls from its queue.
- Traffic weights (`traffic_percent`) live in Postgres. **API processes cache the routing table in memory** and reload it when a message on the Redis pub/sub channel `routing:changed` arrives. There's no Postgres query on the hot path.
- **Sticky routing:** `hash(client_id) % 100` compared against cumulative weights, so a client consistently hits the same version.
- **Rollback:** set the version's weight to 0 and publish. Its queue drains normally, then its workers can be stopped.
- The backpressure check (D6) applies **per queue**.
- Every response carries **`X-Model-Version`**. All metrics are labeled by version.
- *Reconciliation with D8:* several versions can be `live` at once during a canary. The rule is that the `traffic_percent` values of a model's live versions sum to 100, enforced in a transaction when weights change.

> **Changed later:** C1.22 (stable hash with model name; auth before canary; periodic routing reload). See [Part 2](#part-2-design-change-log).

### D10: Response cache
- Redis key: `cache:{model}:{version}:{sha256(canonical_input)}`.
- **No invalidation is ever needed.** A version is immutable, so `(version, input)` always gives the same answer. A new version simply uses new keys.
- The lookup happens **after version routing, before backpressure and enqueue**. Cache hits skip the queue and don't count toward backpressure.
- Inputs are **canonicalized before hashing** (sorted keys, consistent formatting).
- **TTL plus Redis LRU eviction** bound memory only. They're not for correctness.
- Models run in **deterministic eval mode** (no dropout, no sampling).
- Caching is configurable per model (`cache_enabled`).

> **Changed later:** C1.22 (cache in its own Redis instance). See [Part 2](#part-2-design-change-log).

### D11: Authentication and rate limiting
- **API key format:** `sk_<key_id>_<secret>`. Postgres stores `key_id` and the **SHA-256 hash of the secret**, never the plaintext. The full key is shown **once** at creation.
- Why SHA-256 and not bcrypt: keys are 32 random bytes and can't be brute-forced. Slow hashes exist for low-entropy human passwords, and bcrypt per request would only add latency.
- Lookup is by `key_id`, then a **constant-time comparison** of hashes (`hmac.compare_digest`).
- API processes **cache key → client in memory**. **Revocation** publishes to the Redis channel `apikeys:revoked`, which clears the cached entry immediately.
- **Rate limiting:** a token bucket per client in Redis (capacity = per-minute limit, continuous refill), implemented as **one atomic Lua script** so concurrent API processes can't double-spend the last token. Limits are configurable per client.
- Over the limit returns **429** with `Retry-After` and `X-RateLimit-Remaining`.
- **Fail open:** if Redis is unreachable for the limiter, allow the request and log an error.

> **Changed later:** C1.22 (cached API keys expire, so a missed revocation heals). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C1.16 (504 timeout is 500 ms; the no-recovery argument depends on it), C1.19 (the API's readiness includes the reply listener), C1.21 (worker health served from a thread; Docker doesn't act on unhealthy). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C1.20 (custom buckets, ratio-based SLO alert, one API process per scrape target), C1.21 (worker `/metrics` and health from one HTTP thread). See [Part 2](#part-2-design-change-log).

### D14: Deployment (two setups)
**Benchmark setup (local Docker Compose):** the full architecture. Multiple API processes, worker pools per version, Redis, Postgres, MinIO, Prometheus, Grafana, and the Locust load generator. **All performance numbers in the README come from this setup**, with the hardware documented.

**Live demo (free tier, e.g. Render, ~512 MB RAM, sleeps after idle):**
- **Smaller model:** the full transformer plus Python, PyTorch, and two process types won't fit in 512 MB. Use a small distilled model with **dynamic int8 quantization**, and measure its RAM.
- **One container:** the API process and one worker co-located in a single container under a small process supervisor. It's the same code and the same Redis-mediated architecture, just packed into one service.
- **Bake the demo model into the Docker image** as its initial version, so waking from sleep skips the download. The registry upload flow still works.
- **Cold start:** after waking, the service isn't ready until the model loads and warms up. Expose this through `/health/ready`, and have the demo page show a "warming up" state by polling it. Document the expected first-request delay.
- **Managed backing services:** Neon (Postgres), Upstash (Redis, TLS `rediss://`), Cloudflare R2 (objects). **Verify** that the Redis provider supports what we use (Lua `EVAL`, blocking pops, pub/sub), and estimate Redis commands per request against its free quota.
- The demo shows **functionality**, not performance.

> **Changed later:** C0.1 (local MinIO image), C1.15 (benchmarks run natively), C1.22 (measure memory before the 512 MB demo), C1.23 (Compose runs Redis only in the first release; the live demo is future work). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C1.7 (`model_versions.max_queue_depth`); open proposal P1; C1.23 (no
> Postgres in the first-release core; `clients` and `api_keys` return with stretch stage S1). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C1.18 (`replies:*` expiry is set by the worker). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C2.1 (`error` is an object with a code), C2.5 (timestamps are fractional
> milliseconds; `reply_to` restricted to `replies:<32 hex>`). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C0.2, C1.10, C1.23 (SQLAlchemy/asyncpg/Alembic arrive with S1, boto3 with
> the registry). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C1.6 and C1.7 add work to Stages 3 and 5; C1.5 changes the Stage 11 layouts; C1.11 replaces Stage 3's collector and compute threads with one loop; C1.13 makes `max_length` a per-version setting. **C1.23 replaces this build plan** with a reduced first release (core stages, stretch stages, future work). See [Part 2](#part-2-design-change-log).

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

> **Changed later:** C1.11 (the collector/compute split was removed). See [Part 2](#part-2-design-change-log).

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
Full report: [`benchmarks/stage-1-batching.md`](benchmarks/stage-1-batching.md). Every figure
below traces to one of these result files, committed in `bench/results/`:

| Run | Script | Model | Threads | Inputs | Iterations | File |
|---|---|---|---|---|---|---|
| A | `bench_batching.py` | DistilBERT | 8 | ~26 tokens | 100 | `stage1-transformer_classifier-20261106T150229Z.json` |
| S | `bench_batching.py` | sklearn (iris) | n/a | 4 features | 100 | `stage1-sklearn_classifier-20261106T150255Z.json` |
| B | `bench_batching_1.py` | DistilBERT | 4 | ~26 tokens | 300 | `stage1-transformer_classifier-20261106T151131Z.json` |
| C | `bench_batching_1.py` | DistilBERT | 8 | ~26 tokens | 300 | `stage1-transformer_classifier-20261106T151327Z.json` |

### C1.1: Measured batching gain is 4.2×, not ~10× (D1)

- **Before:** "roughly 10× in illustrative numbers."
- **After:** batching 16 requests makes each **4.2×** cheaper (13.6 → 3.2 ms per item; one worker
  goes from 74 to 309 items/s). Gains flatten after 16: going to 32 adds 11% throughput and nearly
  doubles batch time. At the chosen batch size of 8: 3.4×, 253 items/s.
- **Why:** D1 asked for its illustrative numbers to be replaced with measured ones.
- **Evidence:** Table 1 of the report.
  - `bench/results/stage1-transformer_classifier-20261106T151131Z.json` (run B): the 4.2×, 3.4×, 74 / 253 / 309 items/s figures
  - `bench/results/stage1-transformer_classifier-20261106T150229Z.json` (run A) and `bench/results/stage1-transformer_classifier-20261106T151327Z.json` (run C): the same 4.2× with 8 threads

> **Qualified by C1.13:** all these figures are for ~26-token inputs. The gain depends on input length.

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
- **Evidence:** reasoning (Table 4 of the report), with T taken from
  - `bench/results/stage1-transformer_classifier-20261106T151131Z.json` (run B): T = 54.6 ms at batch 16, 33.4 ms at batch 8

> **Superseded by C1.11:** without the collector thread, the saturated case is 2T, not 3T.

### C1.3: "Target load" defined (D1)

- **Before:** "p99 < 150 ms at target load", with target load undefined.
- **After:** target load = **at most 70% of a worker's measured throughput**. The 150 ms p99
  target applies there. Saturation is overload, where 503s (D6) are the intended response.
- **Why:** a latency target needs a defined load to be testable (Stage 11).
- **Evidence:** a definition, not a measurement. The throughput it refers to (253 items/s at
  batch 8) is from
  - `bench/results/stage1-transformer_classifier-20261106T151131Z.json` (run B)

> **Superseded by C1.12:** two guarantees, one for normal load and one for overload.

### C1.4: Main model `max_batch_size` 16 → 8 (D5)

- **Before:** `max_batch_size = 16`, `max_wait = 15 ms`.
- **After:** `max_batch_size = 8`. `max_wait` stays 15 ms until the Stage 11 sweep.
- **Why:** at 16, a saturated worker's worst case is 3T = 164 ms (p99), over 150 ms even with an
  empty queue and before network overhead. At 8 it is 100 ms. Cost: 18% lower peak throughput
  per worker (253 vs 309 items/s).
- **Evidence:** Table 4 of the report.
  - `bench/results/stage1-transformer_classifier-20261106T151131Z.json` (run B): T at batch 16 and 8, and the 253 vs 309 items/s

> **Superseded by C1.14:** `max_batch_size = 4`, once inputs up to 48 tokens were measured.

### C1.5: Inference threads = number of performance cores (D7)

- **Before:** "PyTorch's internal threads set explicitly so total threads never exceed cores";
  benchmark layouts 1×8, 2×4, 4×2.
- **After:** threads per worker × workers = **performance-core count** (4 on the M1). On machines
  where all cores are the same, use the physical core count. Stage 11 layouts on this machine:
  1×4, 2×2, 4×1, plus one 8-thread layout to show the effect under load.
- **Why:** with 8 threads on the M1, each operation waits for the slowest (efficiency) core.
  Compared with 4 threads at batch size 16: median 5% slower, p99 about 2× worse
  (107.8 vs 54.6 ms). With 4 threads, p99 is within 6% of the median.
- **Evidence:** Table 2 of the report.
  - `bench/results/stage1-transformer_classifier-20261106T151131Z.json` (run B): 4 threads
  - `bench/results/stage1-transformer_classifier-20261106T151327Z.json` (run C): 8 threads, same method
  - `bench/results/stage1-transformer_classifier-20261106T150229Z.json` (run A): an independent 8-thread run showing the same tail (p99 88.3 ms at batch 16)

### C1.6: The collector holds at most one formed batch ahead (D7)

- **Before:** "The collector hands full batches to the compute thread through an in-process queue."
- **After:** the collector may hold **at most one formed batch** beyond the one running. It starts
  forming the next batch only once the compute thread has taken the previous one.
- **Why:** batches held inside the worker are invisible to the queue-depth check (D6) and add
  latency. The latency model (C1.2) and the queue limit (C1.7) both assume this bound.
- **Evidence:** reasoning about the worker design; no benchmark run.
- **Implemented in:** Stage 3.

> **Superseded by C1.11:** the collector thread is removed, so no batch is formed ahead.

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
  - `bench/results/stage1-transformer_classifier-20261106T151131Z.json` (run B): T = 33.4 ms (transformer, batch 8)
  - `bench/results/stage1-sklearn_classifier-20261106T150255Z.json` (run S): T = 0.28 ms (sklearn, p99 at batch 1)

> **Amended by C1.11 and C1.14:** the formula's `− 2` becomes `− 1`, and the transformer value is
> recomputed for batch size 4 and 48-token inputs. The per-version setting itself is unchanged.

### C1.8: Why the contrast model doesn't batch (section 2, D5)

- **Before:** sklearn inference "is ~1–2 ms, so batching is turned off ... a case where batching
  isn't worth it."
- **After:** measured cost is a flat **0.12 ms** per call for 1 to 32 items, so batching *does*
  cut per-item cost (15× at 16). It stays off because waiting up to `max_wait` (15 ms) to fill a
  batch costs far more than the 0.12 ms it would save. `max_batch_size = 1` is unchanged.
- **Evidence:**
  - `bench/results/stage1-sklearn_classifier-20261106T150255Z.json` (run S): 0.12 ms per call from 1 to 32 items

### C1.9: Transformer input truncated at 128 tokens (D1, new)

- **Before:** not specified (the model accepts up to 512 tokens).
- **After:** inputs are truncated to `max_length = 128` tokens (configurable).
- **Why:** a batch is padded to its longest input, so one very long input would slow every request
  in its batch. Capping the length bounds the worst-case batch time that C1.2 relies on.
- **Evidence:** reasoning; 128 was not measured when this was decided (it was later, in
  - `bench/results/stage1-transformer_classifier-20261110T095534Z.json` (run F)

> **Superseded by C1.13:** `max_length = 48`, per version and immutable, not global config.

### C1.10: Worker-only dependencies; CPU-only PyTorch (D3, section 7)

- **Before:** "PyTorch (CPU) + Hugging Face transformers" in the tech stack.
- **After:** torch, transformers and scikit-learn are an optional `worker` extra
  (`uv sync --extra worker`), so API installs and images never include them. On Linux, torch comes
  from PyTorch's CPU-only wheel index; PyPI's Linux wheel is the CUDA build with ~3 GB of
  unused NVIDIA libraries.
- **Why:** D3 says API processes never load models; packaging now enforces it.
- **Evidence:** packaging change; no benchmark run.

## Stage 1 follow-up: design audit and input-length runs (2026-11-10)

Two sources: a design review of the whole plan (risks R1 and R2 led to the entries below), and six
more benchmark runs at different input lengths. Same machine and method as Stage 1: Apple M1,
4 threads, model only. New result files in `bench/results/`:

| Run | Words → tokens | Iterations | Batch sizes | File |
|---|---|---|---|---|
| D | 8 → 10 | 300 | 1–32 | `stage1-transformer_classifier-20261110T093840Z.json` |
| E | 96 → 98 | 300 | 1–32 | `stage1-transformer_classifier-20261110T094516Z.json` |
| F | 140 → 128 (truncated) | 300 | 1–32 | `stage1-transformer_classifier-20261110T095534Z.json` |
| G | 46 → 48 | 300 | 1–32 | `stage1-transformer_classifier-20261110T103413Z.json` |
| H | 62 → 64 | 300 | 1–32 | `stage1-transformer_classifier-20261110T104036Z.json` |
| I | 46 → 48 | **1000** | 4, 8 | `stage1-transformer_classifier-20261110T105742Z.json` |

The "Latency budget" block these runs print still uses the old formulas (fixed in Stage 3).
Ignore it; the budgets below are computed by hand from the measured T.

### C1.11: Worker loop pulls a batch only when compute is free (D5, D7; replaces C1.6, amends C1.2, C1.7)

- **Before:** a collector thread builds the next batch while the compute thread runs the current
  one, handing it over through an in-process queue (D7), limited to one formed batch ahead (C1.6).
- **After:** **one loop, no collector thread.** Block for the first request. Take whatever is
  already queued, up to `max_batch_size`, in one Redis call (`LPOP key count`, Redis ≥ 6.2). If the
  batch isn't full, wait for more only until `max_wait` has passed since the first request's
  `enqueued_at_ms`. Run the batch, push the replies, repeat. The worker keeps helper threads only
  for metrics and health.
- **Latency model (replaces C1.2's saturated rows):** with d requests ahead in Redis and
  B = `max_batch_size`,
  ```
  worst case = T × (1 + ceil((d + 1) / B))      # running batch, then the batches up to its own
  m = floor((target − overhead) / T) − 1         # batches from the queue that fit (was − 2)
  max_queue_depth = m × B × workers
  ```
  Light load is still `max_wait + T`. Saturated with an empty queue is **2T** (was 3T).
- **D5 wording:** `max_wait` bounds how long a batch stays open while compute is idle. It does
  not bound how long a request waits for its batch to start under load.
- **Why:**
  - The formed batch is the middle T of C1.2's 3T. It's a full batch of backlog that has left
    Redis, so under load the backlog can never be smaller than one batch, and the `LLEN` check
    (D6) can't see it.
  - The obvious implementation (a handoff queue of size 1) is worse: one batch running, one in the
    queue, one held by the collector blocked on `put()`, giving 4T. Getting C1.6's 3T needed an
    extra rule; this loop needs none.
  - With every waiting request in Redis, the queue limit controls the whole backlog.
  - Less code: one loop instead of two threads, a handoff queue and a rule.
- **Trade-off:** collecting no longer overlaps computing. Under load the requests are already in
  Redis, so collecting costs one round trip. Stage 2 measures it to confirm it is small next to T.
- **Status:** decided 2026-11-10.
- **Evidence:** reasoning from C1.2 (risk R1 of the audit), not yet measured end to end. The T
  values the formula is applied to come from
  - `bench/results/stage1-transformer_classifier-20261106T151131Z.json` (run B)
  - `bench/results/stage1-transformer_classifier-20261110T105742Z.json` (run I)

### C1.12: Latency target as two guarantees (D1, D6; replaces C1.3)

- **Before (C1.3):** the 150 ms p99 target applies at ≤ 70% load; "saturation is overload, where
  503s take over."
- **What broke:** C1.4 chose batch size 8 because batch size 16 misses 150 ms *under saturation*.
  If the target didn't apply under saturation, that reasoning would not hold. The two entries
  contradicted each other.
- **After:**
  1. **Normal operation (≤ 70% of measured worker throughput):** end-to-end p99 < 150 ms, and
     essentially no 503s. 70% leaves room for bursts, since queue wait grows sharply as
     utilization approaches 100%.
  2. **Overload (above that):** the server sheds load with 503 + `Retry-After`. **Requests it
     accepts still complete within 150 ms**, because `max_queue_depth` is derived from that
     budget. That is the purpose of the 503 threshold, and why batch size is chosen for the
     saturated case.
- **Both apply only to inputs up to `max_length` tokens** (C1.13).
- **Evidence:** the budgets behind guarantee 2 use T from
  - `bench/results/stage1-transformer_classifier-20261110T105742Z.json` (run I): batch 4 vs 8 at 48 tokens
- **Open follow-up:** P10 (what "measured throughput" means, and whether guarantee 2 is a target
  that the load test verifies).

### C1.13: Input length is part of the latency target; `max_length` 128 → 48 (D1, D10; replaces C1.9)

- **Before (C1.9):** `max_length = 128`, a global config setting. T was measured only on
  ~26-token inputs.
- **What broke:** a batch pads to its longest input, and at 128 tokens one batch of 8 alone takes
  162.6 ms at p99, over the whole target. The budget held only for short inputs.
- **Findings (batch size 8, 4 threads):**

  | Tokens per input | p50 (ms) | p99 (ms) | Per-item speedup at batch 32 | Run |
  |---|---|---|---|---|
  | 10 | 19.5 | 21.5 | 10.6× | D |
  | 26 | 31.6 | 33.4 | 4.6× | B |
  | 48 | 48.1 | 64.5 | 3.0× | I (G for speedup) |
  | 64 | 62.0 | 76.0 | 2.3× | H |
  | 98 | 92.5 | 109.2 | 1.8× | E |
  | 128 | 122.3 | 162.6 | 1.8× | F |

  - **Batch time tracks total padded tokens, not request count.** Batches with similar totals take
    similar time: 32 × 10 tokens (320) 42 ms, 16 × 26 (416) 52 ms, 4 × 98 (392) 52 ms. Past
    roughly 400 tokens per batch the CPU is fully used, and each extra token costs about 0.12 ms.
  - **So batching helps less as inputs get longer.** Short inputs leave the CPU underused and
    batching fills it; long inputs fill it alone. At 128 tokens throughput is flat at about
    65 items/s from batch size 4 up.
- **After:**
  - The latency target (C1.12) is stated for inputs **up to `max_length` tokens**.
  - **`max_length = 48`**: the longest measured length where a small batch still fits the budget
    (64 tokens fails even at the median, see C1.14).
  - `max_length` is **part of the model version and immutable**, like the weights. Changing it
    changes outputs for the same input (a long text truncated at 48 vs 128 can get a different
    label), which would silently make D10's cached answers stale.
  - Longer inputs are truncated, keeping the start. The API also caps input size in characters,
    since it has no tokenizer (P8).
- **Why 48 is acceptable:** the model was fine-tuned on SST-2, single sentences from movie reviews,
  so long multi-paragraph texts are outside what it was trained on anyway.
- **Rejected:** budgeting for 128 tokens (even one batch of 8 misses 150 ms, and short inputs would
  get needless 503s). **Future work:** token-budget batching, which caps each batch by total
  padded tokens so T stays steady whatever the input length.
- **Unexplained:** at 98 tokens, batch size 16 has p99 = 551 ms and a worse per-item cost than
  batch size 8 (run E). Possibly cache effects on larger tensors, or thermal throttling after
  several runs. It's a configuration we won't deploy, so it is noted rather than investigated.
- **Evidence:**
  - `bench/results/stage1-transformer_classifier-20261110T093840Z.json` (run D): 10 tokens
  - `bench/results/stage1-transformer_classifier-20261106T151131Z.json` (run B): 26 tokens
  - `bench/results/stage1-transformer_classifier-20261110T103413Z.json` (run G): 48 tokens, all batch sizes (incl. the 3.0× speedup)
  - `bench/results/stage1-transformer_classifier-20261110T105742Z.json` (run I): 48 tokens, batch 4 and 8, 1,000 iterations (the 64.5 ms p99)
  - `bench/results/stage1-transformer_classifier-20261110T104036Z.json` (run H): 64 tokens
  - `bench/results/stage1-transformer_classifier-20261110T094516Z.json` (run E): 98 tokens (incl. the unexplained 551 ms p99 at batch 16)
  - `bench/results/stage1-transformer_classifier-20261110T095534Z.json` (run F): 128 tokens

### C1.14: Main model `max_batch_size` 8 → 4; queue limit recomputed (D5, D6; replaces C1.4)

- **Before (C1.4):** `max_batch_size = 8`, chosen on 26-token inputs.
- **What broke:** at 48 tokens, batch-8 p99 is 64.5 ms (run I, 1,000 iterations), so the
  saturated worst case 2T + 30 ms = 159 ms misses the 150 ms target.
- **After:**

  | | Batch 4 | Batch 8 |
  |---|---|---|
  | p50 / p99 at 48 tokens (run I) | 29.2 / 38.9 ms | 48.1 / 64.5 ms |
  | Saturated worst case, 2T + 30 ms | **108 ms** | 159 ms |
  | Throughput, one worker | 137 items/s | 167 items/s |

  - **`max_batch_size = 4`.** `max_wait` stays 15 ms.
  - **`max_queue_depth` = 8 per worker:** m = floor((150 − 30) / 38.9) − 1 = 2, so 2 × 4 = 8.
    Worst case at the limit (7 requests ahead): 3T + 30 = 147 ms.
  - Sklearn under the C1.11 formula: m ≈ 427, still hundreds per worker on model-only timing.
- **Why not stretch to 8:** it misses by only 9 ms, but the 30 ms overhead is still an
  assumption. If measured overhead is 20 ms, batch 8 lands at 149 ms, too thin a margin to derive
  the queue limit from.
- **Trade-off:** 18% less peak throughput per worker (137 vs 167 items/s).
- **Revisit:** after Stage 2 measures real overhead, and in the load-test stage (batch size 4 vs 8
  end to end). Switching is a one-number config change.
- **Note on p99 noise:** run G (300 iterations) gave batch-8 p99 = 89.8 ms at 48 tokens, higher
  than at 64 tokens, which can't reflect real cost. With 300 samples p99 is the 3rd-slowest batch.
  Run I repeated it with 1,000 iterations; its value is the one used.
- **Evidence:**
  - `bench/results/stage1-transformer_classifier-20261110T105742Z.json` (run I): all figures in the table above
  - `bench/results/stage1-transformer_classifier-20261110T103413Z.json` (run G): the noisy 89.8 ms p99 at 300 iterations
  - `bench/results/stage1-transformer_classifier-20261110T104036Z.json` (run H): 64 tokens fails even at the median (62.0 ms)
  - `bench/results/stage1-sklearn_classifier-20261106T150255Z.json` (run S): sklearn T for its queue limit

### C1.15: Benchmarks run natively, not in Docker Compose (D14, C1.5; resolves P5)

- **Before (D14):** "All performance numbers in the README come from [the Docker Compose] setup."
- **What broke:** on macOS, Docker runs containers inside a Linux VM. The same benchmark inside a
  container (48 tokens, 1,000 iterations, `python:3.12-slim`, 8 vCPUs) was about **2.8× slower**:

  | | Native, 4 threads (run I) | Container, 4 threads (run J) | Container, 8 threads (run K) |
  |---|---|---|---|
  | Batch 4: p50 / p99 | 29.2 / 38.9 ms | 80.8 / 159.3 ms | 77.4 / 151.4 ms |
  | Batch 8: p50 / p99 | 48.0 / 64.5 ms | 141.0 / 149.4 ms | 131.6 / 223.9 ms |
  | Batch 4 throughput | 137 items/s | 49 items/s | 52 items/s |

  In the container, even one batch of 4 has p99 over 150 ms, so no queue limit could meet the
  target there, and none of the Stage 1 numbers would apply.
- **Not emulation:** `docker run --rm python:3.12-slim uname -m` printed `aarch64`, so the container
  ran native ARM code.
- **Likely cause (not separated by experiment):** macOS PyTorch most likely does its matrix math
  through Apple's Accelerate library, which can use the M1's matrix hardware that a Linux guest
  can't reach, and the forward pass is 98% matrix multiplies (Table 3 of the report). The VM also
  hides the P/E core split, so its vCPUs may run on efficiency cores, which fits the worse tail.
  4 vs 8 threads is not meaningful inside the VM for the same reason.
- **After:**
  - **All README performance numbers come from a native setup:** API and worker run with `uv run`
    on macOS; only Redis runs in Docker. The hardware is stated with every number.
  - Dockerfiles and the Compose setup are still built (graceful shutdown, restart policy, the
    kill-a-worker demo). They show the system runs in containers; they are not where numbers come
    from.
  - The README states two limitations: the load generator shares the machine, and container
    inference on macOS measured 2.8× slower (with the reason above).
  - C1.5 (threads = performance cores) stays valid, since it was measured in the environment the
    benchmarks run in.
- **Also decided from audit risk R3** (applies to the load-test stage): load must be **open-loop**
  (fixed arrival rate), and README percentiles come from the load generator, not from Prometheus
  histograms. Locust's `constant_throughput` only caps each user's rate and still waits for each
  reply, so it is open-loop only approximately; a fixed-arrival-rate executor (e.g. k6
  `constant-arrival-rate`) is the stricter choice. The tool is picked in that stage.
- **Evidence:**
  - `bench/results/stage1-transformer_classifier-20261110T105742Z.json` (run I): native baseline
  - `bench/results/stage1-transformer_classifier-20261110T121123Z.json` (run J): container, 4 threads
  - `bench/results/stage1-transformer_classifier-20261110T121918Z.json` (run K): container, 8 threads
  - The `uname -m` check above (output `aarch64`).

### C1.16: Request timeout is 500 ms (D6, D12; resolves P6)

- **Before:** D6 sets each request's deadline to "arrival time + client timeout", and D12 returns
  504 "when the handler's timeout fires". Neither gave the timeout a value.
- **After:** one server setting, **`request_timeout_ms = 500`**. It sets both `deadline_ms`
  (arrival + 500 ms) and the API's 504 timer. 503 responses send **`Retry-After: 1`**.
- **Why a timeout is bounded on both sides:**
  - **Not at or near the 150 ms target (e.g. 150 or 200).** The p99 target is statistical: 1% of
    requests may be a little slower. A timeout is a hard cutoff, so setting it at the target turns
    slow-but-fine requests into 504s. The worker drops a request whose deadline passed before
    compute (P9), but one that expires mid-batch is computed and its reply discarded, so the work is
    wasted. Clients then retry, adding load exactly when the server is slow (a retry storm). And
    the 147 ms worst case (C1.14) rests on an assumed 30 ms overhead and on p99s that showed one-off
    spikes, so 200 ms leaves too little margin over an unmeasured number.
  - **Not very high (e.g. 1 s or 30 s).** The queue limit already caps how long an accepted
    request can legitimately take (about 147 ms). A request still waiting far beyond that is not
    slow but broken: a hung or dead worker, or a lost reply. A long timeout only keeps the client
    waiting for that failure. Meanwhile every stuck request holds an HTTP connection and a
    futures-map entry, so a hung worker piles up thousands of them and turns a worker failure
    into an API failure. The client may disconnect first, so the worker's deadline check would
    be computing for clients that are gone.
  - **Rule:** the timeout is the point past which slowness means failure. 500 ms is about 3× the
    worst case an accepted request should take: room for noise and unmeasured overhead, while a
    broken request fails within half a second.
- **Dependency on D12:** D12 skips recovering in-flight requests after a worker crash because
  detecting the crash takes longer than the deadline, so recovered requests would already be
  expired. That holds only while the deadline is short. A multi-second timeout would make
  recovery worth building (Redis Streams with acks).
- **`Retry-After`:** HTTP's `Retry-After` takes whole seconds, so 1 is the smallest value, longer
  than the queue takes to drain. The load-test client's backoff with jitter may start lower.
- **Revisit:** after Stage 2 measures overhead; the measured end-to-end p99 should sit well below
  500 ms (P12).
- **Evidence:** reasoning; no benchmark run. The 147 ms worst case comes from
  `bench/results/stage1-transformer_classifier-20261110T105742Z.json` (run I, via C1.14).

### C1.17: `max_queue_depth` is an explicit setting, computed offline (D6, C1.7; resolves P7)

- **Before (C1.7):** a per-version `max_queue_depth` *defaulting to* the budget formula
  `m × max_batch_size × workers`, with m from T and overhead (C1.11).
- **What broke:** the API enforces the limit, but it knows only two of the formula's five inputs:

  | Input | Who knows it | API can get it? |
  |---|---|---|
  | target (150 ms) | config | yes |
  | `max_batch_size` | per-version config | yes |
  | overhead | nobody at runtime (a Stage 2 benchmark result) | no |
  | T | only the worker, which runs the model (D3) | no |
  | workers | nobody, and it changes on crash or restart | no |

- **After:**
  - `max_queue_depth` is an **explicit per-version setting** holding the **total for that
    version's queue**. There is no computed default.
  - The benchmark script prints the recommended value from the C1.11 formula; the value is set
    from that, by hand.
  - Changing the number of workers for a version means changing this setting. The docs say so.
- **Rejected: computing it live** (workers publish T and heartbeats to Redis; the API recomputes).
  T depends on input length (C1.13), so a burst of long inputs would lower the limit and a burst
  of short ones would raise it: the limit would swing from moment to moment, making it hard to
  test and explain. It also adds heartbeats, staleness rules and a fallback, each a new failure
  mode.
- **How production systems do it:** model servers use a fixed, configured limit tuned by load
  testing, as here: Triton (`max_queue_size` in the dynamic-batching queue policy), TensorFlow
  Serving (`max_enqueued_batches`), TorchServe (`job_queue_size`). General backend services often
  adapt the limit live, but from observed latency, not from T × workers: Netflix's
  `concurrency-limits` library and Envoy's adaptive concurrency filter. Limiting time spent
  waiting rather than queue length is another common approach; the worker's deadline drops (P9)
  are this project's time-based limit, behind the length-based one.
- **Known limitation:** the limit assumes a fixed worker count. If one of two workers crashes, the
  queue drains at half speed while the limit stays the same, so some accepted requests miss
  150 ms. Deadline drops and 504s catch the worst cases, and the worker restarts (D12).
  **Future work:** an adaptive, latency-based limit.
- **Evidence:** reasoning; no benchmark run. Current recommended value (8 per worker for the
  transformer) comes from
  `bench/results/stage1-transformer_classifier-20261110T105742Z.json` (run I, via C1.14).

### C1.18: The worker sets the reply list's expiry (D4, section 6; resolves P11)

- **Before (D4):** "Reply lists get an `EXPIRE` that the listener refreshes periodically, so a dead
  API process's list disappears on its own."
- **What broke:** two Redis facts. A list that becomes empty is deleted, and `EXPIRE` on a key that
  doesn't exist does nothing (it stores no expiry for later).
  - **While the API process is alive**, its listener pops replies almost as soon as they arrive,
    so the list is almost always empty and the key doesn't exist. The periodic refresh almost
    always does nothing.
  - **After the API process dies**, workers still finish its in-flight batches and `RPUSH` their
    replies. The push creates a new key with no expiry. The refresh and the pops both ran in the
    dead process, so nothing ever expires or empties that key: it stays forever, and every crash
    leaves another one.
- **After:** the worker sends `RPUSH replies:{api_instance_id} <reply>` and
  `EXPIRE replies:{api_instance_id} 60` together in one pipeline (one round trip; `MULTI`/`EXEC` if
  it must be atomic). The listener no longer refreshes anything. Rule: **whoever creates the key
  sets its expiry.** 60 s is far above the 500 ms request timeout (C1.16); a reply older than that
  is useless anyway.
- **Rejected: a pub/sub channel per API process.** It stores nothing, so there is nothing to
  clean up, and `PUBLISH` even reports 0 receivers when the process is gone. But pub/sub is
  at-most-once: a reply published while the listener is disconnected (a brief network blip, a
  reconnect) is dropped, and that client gets a 504 after 500 ms and retries. A slow listener
  can also be disconnected by Redis once its output buffer limit is hit, losing more. A list
  holds replies until popped, so after a 200 ms gap the listener catches up and the clients get
  normal, slightly late responses.
- **Rule this follows:** pub/sub where losing a message is harmless, lists where it isn't.
  `routing:changed` and `apikeys:revoked` stay on pub/sub because a periodic reload repairs a
  missed message (P17); replies use lists because a lost reply is a failed request.
- **Section 6 change:** `replies:{api_instance_id}` is "list (expiry set by the worker on each
  push)", not "list (with refreshed EXPIRE)".
- **Evidence:** documented Redis behaviour (empty lists are deleted; `EXPIRE` on a missing key
  returns 0; pub/sub does not store messages); no benchmark run.

### C1.19: The reply listener supervises itself; readiness reflects it (D4, D12; resolves P12)

- **Before (D4):** "Each API process runs one background listener on its reply list and resolves
  an in-memory map `request_id → asyncio.Future`." Nothing said what happens if it fails.
- **What broke:** the listener is one asyncio task looping on `BLPOP`. If Redis restarts, `BLPOP`
  raises a connection error, and if nothing catches it the task ends. Redis never reconnects to
  clients; clients reconnect themselves. redis-py's pool gives ordinary commands a fresh
  connection, so handlers keep enqueueing and workers keep pushing, but no code calls `BLPOP`
  any more. Every request in that API process then waits on a future nobody resolves and gets a
  504; replies pile up in its reply list until they expire (C1.18). `/health/live` still returns
  200, and an exception in an un-awaited task is only logged ("Task exception was never
  retrieved") when the task is garbage-collected. A **silent failure**: worse than a crash,
  because a crash is restarted (D12) and this isn't.
- **After:**
  - **Errors are caught inside the loop; the task never exits.** On a connection error it waits,
    then retries. A single bad message (e.g. malformed JSON) is logged and skipped, so one reply
    can't stop the listener for everyone.
  - **Retries use capped exponential backoff with jitter** (cap about 1 s). Retrying instantly
    would spin the CPU and flood logs, and every API process retrying at once when Redis returns
    could overload it (a thundering herd). The low cap keeps recovery quick.
  - **A dedicated Redis connection** for the listener, since `BLPOP` keeps its connection busy
    while waiting. `BLPOP` timeout 1 s; socket timeout longer (e.g. 2 s), or every idle wait would
    be reported as a timeout error.
  - **Readiness, not liveness.** The loop wakes at least once a second (the `BLPOP` timeout) and
    records a last-iteration timestamp. The API's `/health/ready` is false when it is older than
    3 s, which catches both a crashed and a stuck listener. `/health/live` stays
    dependency-free: not live means "restart me", which can't fix Redis and would cause a restart
    loop; not ready means "stop sending me traffic", and traffic returns once the listener
    recovers.
  - **Futures-map entries are removed in `finally`** in the handler, so success, timeout (504),
    a failed enqueue and cancellation all clean up. Removing it only on timeout would leak an
    entry per request whenever the enqueue fails, growing fastest during an outage. Use
    `pending.pop(request_id, None)`, which doesn't raise if the entry is already gone. Late
    replies then find no entry and are discarded (D4).
  - **The listener checks `fut.done()` before `set_result`.** A handler can time out at the same
    moment its reply arrives; `set_result` on a finished or cancelled future raises
    `InvalidStateError`, which, uncaught, would kill the listener.
- **Evidence:** documented redis-py and asyncio behaviour; no benchmark run.

### C1.20: Latency buckets around the target; one API process per scrape target (D13; resolves P13)

- **Before (D13):** histograms with no bucket boundaries specified, so `prometheus_client`'s
  defaults; "API processes expose `/metrics` via FastAPI"; alert on "p99 > 150 ms for 5 minutes".
- **What broke (buckets):** a Prometheus histogram keeps, per bucket, a counter of requests that
  took at most that long (since process start). Prometheus scrapes the counters; the time window
  is chosen at query time, e.g. `histogram_quantile(0.99, rate(latency_bucket[5m]))`, which
  assumes requests are spread evenly inside a bucket. The default boundaries jump from 0.1 s to
  0.25 s, so a 150 ms target sits inside one 150 ms-wide bucket. Example: 1,000 requests, 900 at
  ≤ 100 ms and 100 between 100 and 250 ms gives an estimated p99 of 100 + 0.9 × 150 = 235 ms,
  whether those 100 requests took 140 ms or 160 ms. The alert could neither tell 140 from 160 nor
  report the real value. Grafana's p99 graph is computed from the same buckets, so it has the
  same blind spot.
- **What broke (processes):** with `uvicorn --workers N`, N processes share one port and each
  keeps its own counters. A scrape reaches whichever process the OS picks, so Prometheus sees one
  process's share (e.g. 600, 400, 610, 405), never the total. `rate()` reads each drop as a
  counter reset after a restart and counts the full value again, filling graphs with false spikes.
- **After:**
  - **Custom buckets** for latency histograms:
    `0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.5, 1` s, with a boundary exactly
    at the 150 ms target.
  - **The SLO alert uses an exact ratio, not the interpolated p99:**
    `rate(latency_bucket{le="0.15"}[5m]) / rate(latency_count[5m]) < 0.99`. Both are exact counts,
    so no interpolation is involved. The interpolated p99 stays on dashboards.
  - **README percentiles come from the load generator** (as in C1.15).
  - **One API process per instance**, each scraped as its own target and summed at query time
    with `sum(...)`. No `--workers N`. Natively (C1.15), the load tests run two API processes on
    two ports, and the load generator spreads requests across both.
- **Rejected: `prometheus_client` multiprocess mode** (each process writes metrics to files in a
  shared directory; `/metrics` merges them). It works, but needs the directory cleared at startup
  and a merge mode for gauges, and some features don't support it. One process per scrape target
  is simpler and matches how containers are scaled.
- **Evidence:** documented Prometheus and `prometheus_client` behaviour; no benchmark run.

### C1.21: Worker health served from a thread inside the worker; one stage owns the containers (D12, D13, section 8; resolves P14)

- **Before:** D12 gives the worker a `/health/ready` (true only after the model is loaded, its hash
  verified and a warm-up batch run). D13 has workers expose `/metrics` with `prometheus_client`'s
  `start_http_server`. The Stage 0 doc says app containers arrive "Stage 9 onward".
- **What broke:**
  - The worker is a plain Python loop, not a web server, and `start_http_server` serves metrics
    only, so `/health/ready` had nowhere to live.
  - No stage's description includes writing the Dockerfiles.
- **Why not a separate health process:** whether the model has loaded and when the loop last ran
  exist only as variables in the worker process's memory. Another process can't read them without
  extra machinery, and loading its own copy of the model would double the RAM (against D3) while
  saying nothing about the worker's copy or its loop.
- **After:**
  - **The worker runs a small HTTP server in a background thread** of the same process, serving
    `/metrics`, `/health/live` (always 200) and `/health/ready`. It shares memory with the main
    loop, so it reads the loop's flags directly.
  - **Worker readiness:** model loaded, hash verified, warm-up batch run (a `ready` flag set by the
    loop), **and** the main loop ran within the last 3 s (a `last_loop_at` timestamp updated every
    iteration; the loop's blocking pop has a 1 s timeout, so it wakes at least once a second even
    when idle). Same pattern as the API listener in C1.19.
  - **Cost:** one mostly idle thread. A health check reads two variables, so it holds the GIL for
    microseconds, and the forward pass (98% of batch time, Table 3 of the benchmark report)
    releases the GIL anyway.
  - **Why this thread is fine when the collector thread (C1.11) wasn't:** the collector held
    requests, forming batches early, which hid backlog from `LLEN` and added a batch of latency.
    The health thread holds no requests; it only reads flags.
  - **One stage owns the Dockerfiles** (API and worker images) and the Compose services: the
    containers-and-reliability stage of the revised plan (P4).
  - **Compose healthchecks use Python,** since `python:3.12-slim` has no `curl`:
    `python -c "import urllib.request; urllib.request.urlopen('http://localhost:9101/health/ready')"`.
- **Documented limitation:** plain Docker does not act on "unhealthy". `restart: unless-stopped`
  restarts a container only when its process exits; an unhealthy one keeps running. Compose uses
  health status only for `depends_on: condition: service_healthy`. Orchestrators such as
  Kubernetes act on liveness and readiness. Locally, health checks give visibility and startup
  ordering, not automatic recovery.
- **Evidence:** documented `prometheus_client` and Docker behaviour; no benchmark run.

### C1.22: Fixes for the routing, invalidation, cache and demo designs (D9, D10, D11, D14; resolves P17)

These apply to stages the revised plan may defer (P4). Recording them now keeps those stages from
rediscovering the problems; it does not commit to building them.

**R10a. Sticky routing uses a stable hash (D9).**
- **Before:** `hash(client_id) % 100`.
- **What broke:** Python mixes a random per-process seed into string hashes (a defence against
  hash-flooding attacks on dictionaries). Two API processes compute different buckets for the same
  client, so its requests flip between versions depending on which process answers. Fixing the
  seed with `PYTHONHASHSEED` would work only if every process set it; forgetting it once breaks
  routing silently.
- **After:** `bucket = int.from_bytes(sha256(f"{model}:{client_id}").digest()[:8], "big") % 100`,
  identical in every process, on every machine and in every language. Buckets are compared with
  cumulative weights (v1 90%, v2 10%: buckets 0–89 → v1, 90–99 → v2).
- **Model name in the hash:** each model's canary group is an independent slice of clients, so the
  same few clients don't test every new version of every model.
- **Canary range grows from the top** (10% = 90–99, 20% = 80–99), so clients already in the canary
  stay there as it grows and nobody flips between versions during a rollout.

**R10b. Auth is built before canary routing; canary tests use many clients (D9, D11, section 8).**
- **What broke:** routing needs `client_id`, which first exists with API keys (old Stage 8), but
  canary routing was old Stage 6. Alternatives are worse: a client-supplied `X-Client-Id` lets a
  client choose its own bucket, and an IP address changes (mobile networks) or is shared by many
  users (NAT).
- **After:** if canary routing is built, authentication comes first, and the routing key is the
  authenticated `client_id`.
- **Testing:** each client occupies one bucket, so with few clients the canary's real share is
  coarse (3 clients and a 10% canary give 0% or 33%). A canary load test uses a few hundred API
  keys, one per simulated user.

**R11. Pub/sub invalidations are backed by periodic reloads (D9, D11).**
- **What broke:** pub/sub is at-most-once (C1.18). An API process that misses `routing:changed`
  keeps a stale routing table forever; one that misses `apikeys:revoked` keeps accepting a revoked
  key forever.
- **After:** API processes also reload the routing table periodically (e.g. every 30 s), and cached
  API keys expire (e.g. after 30 s). Pub/sub still makes changes fast in the normal case; the
  reload bounds staleness to about 30 s when a message is missed.

**R12. The response cache gets its own Redis instance (D10).**
- **What broke:** `maxmemory-policy` applies to a whole Redis instance. `allkeys-lru` can evict
  request queues and rate-limit buckets. `volatile-lru` evicts only keys with an expiry, and since
  C1.18 reply lists have one, so replies could be evicted before the listener pops them.
- **After:** a second Redis instance holds only `cache:*` keys, with `maxmemory` and LRU eviction.
  The main instance keeps no eviction policy.

**R13. Measure memory before committing to the 512 MB demo (D14).**
- **What broke:** D14 assumes Python, PyTorch, a quantized model and two processes fit in about
  512 MB. Not measured.
- **After:** measure the resident memory of the API and worker processes (with the demo model)
  before choosing the free tier. If they don't fit, the demo needs a bigger instance or a smaller
  runtime.

- **Evidence:** documented Python, Redis and HTTP behaviour; no benchmark run.

### C1.23: Scope of the first release and the revised build plan (section 8; resolves P4)

- **Before (section 8):** 13 stages, 0 to 12, all to be built.
- **Why it changed:** 11 stages remained after Stage 1, which would take weeks to build well,
  while the goal is to start applying for backend roles soon. The first release builds what
  backend interviews probe most (concurrency, queues, timeouts, failure behaviour, load shedding,
  measured results) and lists the rest as future work with its design already written.
- **After:** the build plan below **replaces section 8** for implementation. Same deliverables per
  stage (code, tests, `docs/stages/NN-<name>.md`, commit message, verify commands), one stage per
  response.
- **Consequences:** models and versions come from config (`sentiment` v1, `iris` v1), not a
  registry. Queue names stay `q:{model}:{version}`, so the registry and canary routing can be added
  later without renaming. MinIO leaves Compose; Postgres returns only with S1. The README claims
  only what is built.

**Core (ModelDock is resume-ready when these are done)**

| Stage | Contents | Decisions it implements |
|---|---|---|
| 1.5 Housekeeping | Commit `bench/results/` (remove it from `.gitignore`); delete `temp.py`; remove MinIO (and Postgres, until S1) from Compose. | C1.15, C1.23 |
| 2 Queue round trip | Predict endpoint. Input validation per runner kind, limits from per-version config, input-size cap. `request_timeout_ms = 500` sets `deadline_ms` and the 504 timer. Futures map cleaned in `finally`; listener checks `fut.done()`. Self-supervising listener on a dedicated connection; API `/health/ready`. Worker pops one request, checks its deadline, replies with `RPUSH` + `EXPIRE 60` in one pipeline. Reply carries an error code. Then **measure overhead** (end-to-end minus `inference_ms`, p50 and p99). | C1.16, C1.18, C1.19; P3, P8, P9 |
| 3 Batching | Pull-when-ready loop: block for the first request, `LPOP key count` up to `max_batch_size = 4`, wait for more only until `max_wait` (15 ms) after the first request's `enqueued_at_ms`. `max_length = 48` per version. Deadline check just before compute. A failed batch replies with an error to every request in it. Thread caps for PyTorch and numpy/BLAS. Timings in the reply. Benchmark script prints the C1.11 budget and the recommended `max_queue_depth`. | C1.11, C1.13, C1.14, C1.17; P9, P15 |
| 4 Backpressure | Explicit per-version `max_queue_depth` (total for the queue), 503 with `Retry-After: 1`, counter for expired requests. Recompute the queue limit from the overhead measured in Stage 2. | C1.16, C1.17 |
| 5 Containers and reliability | Dockerfiles for API and worker; both in Compose. Worker health/metrics HTTP thread (`/metrics`, `/health/live`, `/health/ready` with the 3 s loop check). SIGTERM handling in both (blocking pops use a 1 s timeout so the loop notices). A script that kills a worker under load. Containers are not used for benchmark numbers. | C1.15, C1.21; D12 |
| 6 Metrics and logs | D13 metrics with custom buckets (boundary at 0.15 s); SLO ratio alert rule; JSON logs with `request_id` in both process types; Prometheus scraping each API process and the worker as separate targets. | C1.20 |
| 7 Load tests | Native setup (two API processes on two ports, worker, Redis in Docker). Open-loop load; client with exponential backoff and jitter on 503. Results: batching on vs off, `max_batch_size` 4 vs 8, a `max_wait` sweep, overload behaviour. Define target load from the measured end-to-end saturation throughput. Percentiles from the load generator. | C1.12, C1.14, C1.15; P10, optional P16 |
| 8 README | Architecture, decisions (linking this document), results with hardware, limitations (shared machine, container slowdown, unhealthy containers not restarted, fixed queue limit on worker crash), future work. Resume bullets from measured numbers only. | all |

If a core stage overruns, drop the `max_wait` sweep from Stage 7 before dropping anything else.

**Stretch (after the core, in this order)**

| Stage | Contents | Decisions |
|---|---|---|
| S1 Rate limiting | Postgres back in Compose; `clients` and `api_keys` tables; key format with hex-only parts (so `sk_<key_id>_<secret>` splits unambiguously); SHA-256 + constant-time compare; key cache with expiry and `apikeys:revoked`; Lua token bucket; 429 headers; fail open. | D11, C1.22 (R11) |
| S2 Response cache | Separate Redis instance with `maxmemory` + LRU; canonical input hash; TTL; one hit-rate benchmark. | D10, C1.22 (R12) |
| S3 Grafana | One dashboard provisioned from files, one row per latency stage. | D13 |

**Future work (designed, not built in the first release)**

- Model registry and object storage (old Stage 5), with P1 (runner type) and P2 (tar artifacts).
- Canary routing (old Stage 6), with C1.22's stable hash, auth first and periodic reload.
- Live demo (old Stage 12), after measuring memory (C1.22, R13).
- Process × thread layout comparison (C1.5).
- Token-budget batching for longer inputs (C1.13).
- An adaptive, latency-based queue limit (C1.17).

- **Evidence:** scope judgment from the design audit, accepted 2026-11-20; no benchmark run.
- **Implemented in:** Stage 1.5 ([`stages/01.5-housekeeping.md`](stages/01.5-housekeeping.md)).
  Compose runs Redis only. The `postgres_dsn` and `s3_*` settings are removed too, since nothing
  reads them; each returns with the stage that uses it (S1, the registry), as `SecretStr` where it
  holds a credential (C0.2). `bench/results/` is committed, and `tests/test_evidence.py` fails
  if any doc cites a result file that is not in the repo (rule 3); new result files also record
  the script and git commit that produced them. `artifacts/` is git-ignored.

## Stage 2: queue round trip

Sources: the Stage 2 proposals (P3, P8, P9, P18), decided 2026-11-22, and gaps found while
building the endpoint. Details in [`stages/02-queue-round-trip.md`](stages/02-queue-round-trip.md).

### C2.1: Inputs validated per model version; error codes; a failed batch fails only its own requests (D3, section 6; resolves P3)

- **Before (P3):** one malformed input in a batch fails the whole batch; proposal: validate in the
  API against a per-model schema, reply to a failed batch with an error rather than letting it
  time out. Section 6's reply had `"error": null` or, implicitly, a string.
- **What broke:**
  - The API never loads a model (D3), so it can't ask the model what a valid input is. The sklearn
    feature count exists only inside `model.joblib`.
  - "Invalid input" can't be recognised from an exception's type alone: numpy and torch raise
    plain `ValueError` for their own internal reasons. Mapping every `ValueError` to 422 would tell
    a client "your request is wrong, don't retry" for a failure that was ours.
  - Replying to a failed batch with one error for every request makes one bad input fail the
    requests batched with it, which is the problem P3 set out to fix.
- **After:**
  - **A model catalog in code** (`common/models.py`): per version, the runner kind, the artifact
    directory and the input limits (`max_chars` for text, `n_features` for features). Structured,
    and it changes only with code or artifacts; deployment knobs stay in environment variables. Part
    of the version, so immutable like the weights (C1.13).
  - **The API validates before enqueueing**, with a pydantic schema built from the version's
    limits: strict types (no `"5.1"` → 5.1, no booleans as numbers), finite numbers (Python's JSON
    parser accepts `NaN`), no unknown fields. Invalid input gets **422** and is never queued.
  - **Drift check:** the worker compares the catalog's `n_features` with the loaded artifact at
    startup and refuses to start if they differ.
  - **`InvalidInputError`** (`worker/runners/base.py`): runners raise it, after checking the input
    themselves, for input they reject. It subclasses `ValueError` and carries the index of the bad
    input in the batch. **Only it** maps to `invalid_input` / 422. Every other exception, including
    a plain `ValueError` from numpy or torch, is `inference_failed` / 500, with a generic message to
    the client and the traceback in the worker's log.
  - **Reply `error` is an object:** `{"code": "invalid_input" | "inference_failed", "message"}`.
  - **The worker keeps running** after any per-request failure.
  - **Stage 3 rule for a failed batch:** validate each input before forming the batch; if the batch
    still fails, re-run each request alone (checking its deadline first) and reply to each with its
    own result. A request gets 500 only if it fails on its own as well, so if every request fails
    alone, all get 500.
- **Why:** 422 and 500 tell the client different things (fix the request vs retry), so the code
    must say whose fault it was. Validating in the API stops bad input before it can share a batch.
    The catalog is the only place both processes can read limits from without the API loading a
    model.
- **Evidence:** reasoning; behaviour pinned by `tests/test_models.py`, `tests/test_worker.py`,
  `tests/test_api_predict.py`, and the runner tests. No benchmark run.

### C2.2: Two input-size limits: body bytes (413) and characters (422) (section 3, C1.13; resolves P8)

- **Before (P8):** about 250 characters as a proxy for 48 tokens, reject (413) or truncate, to be
  decided.
- **What broke:**
  - 250 characters would reject texts the worker handles fine: longer inputs are truncated to
    48 tokens by the tokenizer (C1.13), which is the limit that bounds compute.
  - A field limit doesn't bound request size or JSON parsing: FastAPI reads and parses the whole
    body before any field is checked.
- **After:** two limits, each with one job:
  - **Body bytes, `max_body_bytes = 16384`, → 413**, in raw ASGI middleware (`api/limits.py`). With
    a `Content-Length` header, an oversized request is rejected before a byte of the body is read;
    a chunked body is counted as it arrives and rejected once past the limit. Protects memory and
    parsing.
  - **Characters, `max_chars = 1000` per version, → 422**, in the input schema (C2.1). Protects the
    Redis message and the tokenizer, which processes the whole string before truncating.
  - **Reject, don't truncate,** at the API: truncating would change the client's input without
    telling it. The 48-token truncation in the worker stays (documented behaviour, Stage 3).
- **Sizing:** 1,000 is about 4× a 48-token English text. 16 KB is above the largest valid
  sentiment body: the worst case is 1,000 characters that JSON escapes as 12 bytes each (an emoji,
  a surrogate pair), 12,012 bytes, so the body limit never rejects input the schema would accept.
- **Evidence:** reasoning; the 12,012-byte worst case is computed and checked in
  `tests/test_api_predict.py::test_largest_valid_text_fits_the_body_limit`. No benchmark run.

### C2.3: Deadline checks; timestamps are wall clock (D6; resolves P9)

- **Before (P9):** check the deadline at pop and again just before compute; a dropped request
  gets no reply, only a counter.
- **After:** as proposed. In Stage 2 the worker computes right after popping, so the two checks
  are the same check; Stage 3's batching wait separates them. A dropped request is logged with its
  `request_id` and counted (a plain counter until Prometheus in Stage 6). No reply: the API's timer
  and `deadline_ms` come from the same clock reading and the same 500 ms (C1.16), so the client has
  already had its 504.
- **Known limitation: cross-process timestamps assume synchronised clocks.** `deadline_ms` and
  `enqueued_at_ms` are wall-clock times written by the API and read by the worker. Wall clock
  because monotonic clocks can't be compared between processes (each has its own zero point). On
  one machine the clocks are the same. Across hosts they agree only as well as NTP keeps them,
  typically within milliseconds, and a wall clock can jump when NTP steps it.
  - A worker clock **ahead** of the API's drops requests early and inflates `queue_wait_ms`.
  - A worker clock **behind** computes requests that have already expired: wasted work, but no
    wrong answers, because the API's 504 timer runs on its own monotonic clock (asyncio's loop
    time), not on `deadline_ms`.
  - **Stage 3** also uses `enqueued_at_ms` for `max_wait` (keep a batch open until
    `enqueued_at_ms + max_wait`) and for `queue_wait_ms`. Skew shifts both: a worker clock 10 ms
    ahead leaves a 15 ms `max_wait` only 5 ms. Stage 3 should decide whether to measure `max_wait`
    from the worker's own (monotonic) pop time instead, which skew can't affect.
- **Evidence:** reasoning; documented clock behaviour. No benchmark run.

### C2.4: Redis URLs are secrets (C0.2; resolves P18)

- **Before:** `redis_url` was a plain `str`, though a managed Redis URL carries a password
  (`rediss://default:<password>@host:port`) and C0.2 says credentials are `SecretStr`.
- **After:** `redis_url` is a `SecretStr`, unwrapped only where a client is created (the API's
  lifespan, the worker's `main`). The response cache's separate instance (stretch S2, C1.22 R12)
  gets a `cache_redis_url`, also a `SecretStr`. The secret-masking test is back.
- **Evidence:** `tests/test_config.py::test_redis_url_is_masked_when_printed`.

### C2.5: The HTTP contract and wire-format details (D2, D6, section 6; new)

Not specified before Stage 2; decided while building it.

- **Request:** `POST /v1/models/{name}/predict` with the input itself as the body
  (`{"text": ...}` or `{"features": [...]}`). The model's serving version is chosen by the server
  (one per model now; canary routing later).
- **Success (200):** `{"request_id", "model", "version", "output"}`, with headers `X-Request-Id`,
  `X-Model-Version` (D9) and `Server-Timing` (`app`, `queue`, `inference` durations in ms, a
  standard header that the overhead benchmark reads).
- **Errors** all have one shape, `{"error": {"code", "message", ...}}`, with `details` for schema
  errors (location and message only, never the client's input echoed back) and `request_id` once
  the request was enqueued:

  | Status | `code` | When |
  |---|---|---|
  | 404 | `unknown_model` | no such model |
  | 413 | `body_too_large` | body over `max_body_bytes` (C2.2) |
  | 422 | `invalid_input` | bad JSON, schema failure (C2.1), or the runner's `InvalidInputError` |
  | 500 | `inference_failed` | any other inference exception (C2.1) |
  | 503 | `backend_unavailable` | the enqueue failed (Redis unreachable); `Retry-After: 1` |
  | 504 | `timeout` | no reply within `request_timeout_ms` (C1.16) |

- **503 when the queue is unreachable.** D6 reserves 503 for "the server is overloaded". A Redis
  outage is a different cause with the same meaning for the client: the service can't take the
  request now, retry later. HTTP defines 503 for exactly that, so it is used for both, and the
  `code` tells them apart (Stage 4's overload 503 gets its own code). The alternative, 500, would
  tell clients the failure is a bug rather than a temporary condition.
- **Timestamps are fractional milliseconds** (`deadline_ms`, `enqueued_at_ms`). On one machine the
  whole path takes a few milliseconds, and whole-millisecond values rounded `queue_wait_ms` to 0
  or 1, making the overhead breakdown wrong by up to 1 ms. Seen in a smoke run while building this
  stage (not a recorded benchmark).
- **`reply_to` must match `replies:<32 hex>`**, checked by the worker when it decodes a request. A
  forged or buggy message therefore can't make the worker push into any other key, such as a
  request queue.
- **Unknown message fields are ignored, missing required ones are errors,** so a newer process can
  add a field without breaking an older one during a rolling restart.
- **Queue order is FIFO:** the API `RPUSH`es, the worker pops from the head (`BLPOP` now,
  `LPOP key count` in Stage 3).
- **Evidence:** reasoning; behaviour pinned by `tests/test_api_predict.py`, `tests/test_protocol.py`,
  `tests/test_body_limit.py`, `tests/test_roundtrip_redis.py`.

### C2.6: Overhead is measured at light load and treated as a lower bound (C1.14, C1.16, C1.17, P10)

- **Before:** C1.14's queue limit, C1.16's timeout margin and C1.17's recommended
  `max_queue_depth` all assume 30 ms of overhead (everything outside inference), not measured.
- **After:**
  - `scripts/bench_overhead.py` sends requests **one at a time** through the full path (HTTP, API,
    Redis, worker, reply listener) for each model, and reports p50/p90/p99 of end-to-end time,
    inference time, and **overhead = end-to-end − inference**, split into HTTP (client to handler)
    and RPC (inside the handler, minus queue wait and inference). The sklearn model's inference is
    a fraction of a millisecond, so its end-to-end time is almost pure overhead: a cross-check.
  - **The result is a lower bound.** With one request in flight nothing queues; under load the
    event loop, Redis and the worker add waiting, so overhead grows. A queue limit derived from the
    light-load number is therefore optimistic.
  - **Stage 4** uses it provisionally to recompute `max_queue_depth` (C1.17). **Stage 7's load
    test** checks whether accepted requests under overload really finish within 150 ms
    (guarantee 2, C1.12), which is what P10 asks.
- **Result:** not yet run. The figures and their result files
  (`bench/results/stage2-overhead-{iris,sentiment}-*.json`) are added in a follow-up commit, once
  the benchmark has been run natively on the M1.
- **Evidence:** pending.

---

# Part 3: Current values at a glance

| Parameter | Initial design | Current | Change |
|---|---|---|---|
| Main model `max_batch_size` | 16 | **4** | C1.4, C1.14 |
| Main model `max_wait` | 15 ms | 15 ms | |
| Main model input length | unspecified | **48 tokens, per version, immutable** | C1.9, C1.13 |
| Worker loop | collector thread + compute thread | **one loop, pulls a batch when compute is free** | C1.6, C1.11 |
| Saturated worst case (empty queue) | not modelled | **2T** | C1.2, C1.11 |
| Inference threads (M1) | ≤ cores (8) | **4 (performance cores)** | C1.5 |
| Queue limit | 16 × workers | **`max_queue_depth`: explicit per-version total, set from the benchmark's recommendation (8 per worker for the transformer, provisional until overhead is measured)** | C1.7, C1.11, C1.14, C1.17 |
| p99 target | < 150 ms at target load | < 150 ms for inputs ≤ 48 tokens: **at ≤ 70% load, and for accepted requests under overload** | C1.3, C1.12 |
| Contrast model `max_batch_size` | 1 | 1 (rationale corrected) | C1.8 |
| Benchmark environment | Docker Compose | **Native macOS (Redis in Docker), open-loop load** | C1.15 |
| Request timeout (deadline and 504) | unspecified | **500 ms**; 503s send `Retry-After: 1` | C1.16 |
| Reply-list expiry | refreshed by the API's listener | **set by the worker on each push (60 s)** | C1.18 |
| API readiness | not specified for the API | **false if the reply listener's last loop is older than 3 s** | C1.19 |
| Latency SLO alert | p99 > 150 ms (interpolated) | **fraction of requests ≤ 150 ms < 0.99, from a bucket at exactly 0.15 s** | C1.20 |
| API processes | multiple per port | **one per instance, each scraped separately** | C1.20 |
| Scope | 13 stages, all built | **7 core stages + housekeeping; 3 stretch; registry, canary and demo deferred** | C1.23 |
| Worker health endpoints | `/health/ready` with no server to host it | **HTTP thread in the worker: `/metrics`, `/health/live`, `/health/ready`** | C1.21 |
| Backing services in Compose | Redis, Postgres, MinIO | **Redis only**; Postgres returns with S1, MinIO (Chainguard build) with the registry | C0.1, C1.23 |
| Input validation | none | **per-version schema in the API (422 before enqueue); runner `InvalidInputError` → 422, any other exception → 500** | C2.1 |
| Input size limits | unspecified | **body ≤ 16 KB (413), text ≤ 1,000 characters (422), then truncation to 48 tokens in the worker** | C2.2, C1.13 |
| Redis URL | plain string | **`SecretStr`** (and the future `cache_redis_url`) | C2.4 |
| Wire timestamps | integer ms | **fractional ms, wall clock (assumes synchronised clocks across hosts)** | C2.3, C2.5 |
| Serving overhead | 30 ms assumed | **30 ms assumed; light-load measurement (a lower bound) pending** | C2.6 |
| Benchmark result files | git-ignored | **committed in `bench/results/`; a test checks every cited file exists; new files record script and git commit** | C1.23 |

---

# Part 4: Open proposals

Not yet decided. When one is agreed, it moves to Part 2 under the stage that implements it.

- **P1: runner type in the registry (section 5; deferred with the registry, C1.23).** The schema has no field saying which
  runner loads a model (transformer vs sklearn). Proposal: a `runtime` column on `models`
  (`transformer_classifier` / `sklearn_classifier`).
- **P2: model artifacts as one archive (D8; deferred with the registry, C1.23).** D8's example key
  `models/sentiment/v3-<hash>.pt` assumes one file, but a saved Hugging Face model is a directory
  (`config.json`, `model.safetensors`, tokenizer files). Proposal: store each version as one
  `.tar`, so the immutable key, SHA-256 check and hash-keyed cache work unchanged.
- **P3: per-model input validation in the API.** Decided: see C2.1.

The proposals below come from the design audit of 2026-11-10. Risks R1 and R2 are decided
(C1.11, C1.13, C1.14). The rest are not yet reviewed.

- **P4: scope.** Decided: see C1.23.
- **P5: where benchmark numbers come from.** Decided: see C1.15.
- **P6: request timeout.** Decided: see C1.16.
- **P7: `max_queue_depth` is explicit.** Decided: see C1.17.
- **P8: input size cap at the API.** Decided: see C2.2.
- **P9: deadline checks.** Decided: see C2.3.
- **P10: target-load definition (C1.12).** The only measured throughput is model-only and native.
  Proposal: define 70% against end-to-end saturation throughput from the load test, and treat
  guarantee 2 as a target the load test verifies, since the bound adds p99s and the queue limit is
  soft.
- **P11: reply-list expiry.** Decided: see C1.18.
- **P12: listener failure.** Decided: see C1.19.
- **P13: metrics.** Decided: see C1.20.
- **P14: containers and worker health.** Decided: see C1.21.
- **P15: thread caps for numpy/BLAS (D7).** Set the thread environment variables before numpy is
  imported, so total threads stay within the performance cores.
- **P16: sklearn batching (C1.8).** C1.8's reason argues for `max_wait_ms = 0`, not for
  `max_batch_size = 1`. Proposal: try `max_wait_ms = 0` with a larger batch size in the load test.
- **P17: recorded for deferred stages.** Decided: see C1.22.
- **P18: `redis_url` as a secret.** Decided: see C2.4.
