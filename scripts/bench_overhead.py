"""Measure the serving overhead: end-to-end latency minus inference time, at light load.

Start Redis, one API process and one worker for the model, then:

    uv run python -m scripts.bench_overhead --model iris
    uv run python -m scripts.bench_overhead --model sentiment

It sends requests **one at a time** (each waits for the previous reply), so there is never a
queue: what's left after subtracting inference is the fixed cost of the path itself.

    client ── HTTP ──► API handler ── RPUSH ──► Redis ── BLPOP ──► worker ── inference
       ▲                    ▲                                         │
       └──── HTTP ──────────┴──── listener ◄── BLPOP ◄── Redis ◄── RPUSH + EXPIRE

Per request it records, from the client's clock and the ``Server-Timing`` header:

- ``e2e_ms``: client-side, request sent to response read (perf_counter)
- ``inference_ms``: inside the worker, around ``predict_batch``
- ``overhead_ms = e2e_ms - inference_ms - batch_ms``: the number C1.14, C1.16 and C1.17 assumed
  was 30 ms. ``batch_ms`` (time in a forming batch, from Stage 3) is subtracted too: with one
  request in flight, a batching model's request waits out the whole ``max_wait`` window, which
  is batcher configuration (D13 measures it as its own stage), not path overhead
- split of the overhead: ``http_ms = e2e - app`` (client, HTTP, uvicorn, middleware) and
  ``rpc_ms = app - queue_wait - inference`` (validation, JSON, Redis round trips, worker decode,
  listener wake-up)

**This is a lower bound** (C2.6). Under load the same path also queues for the event loop,
Redis, and the worker, so overhead grows. A queue limit derived from this number is optimistic;
Stage 4 uses it provisionally and the Stage 7 load test checks the result (P10).

The sklearn model's inference is ~0.1 ms, so its e2e is almost pure overhead: a cross-check
for the transformer's number, which subtracts a much larger inference time.
"""

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from scripts.bench_batching import machine_info, make_iris_features, make_texts, percentile
from scripts.provenance import provenance

LOWER_BOUND_NOTE = (
    "Light load, one request in flight: a lower bound on overhead. Overhead grows under load; "
    "a queue limit derived from this is optimistic until the Stage 7 load test checks it (C2.6)."
)


def parse_server_timing(header: str) -> dict[str, float]:
    """``"app;dur=1.2, queue;dur=0.3"`` -> ``{"app": 1.2, "queue": 0.3}``."""
    out = {}
    for part in header.split(","):
        name, _, params = part.strip().partition(";")
        for param in params.split(";"):
            key, _, value = param.strip().partition("=")
            if key == "dur":
                out[name] = float(value)
    return out


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def run(client: httpx.Client, model: str, inputs: list[dict], warmup: int) -> dict[str, Any]:
    url = f"/v1/models/{model}/predict"
    samples: dict[str, list[float]] = {
        k: []
        for k in (
            "e2e_ms",
            "inference_ms",
            "overhead_ms",
            "http_ms",
            "rpc_ms",
            "queue_ms",
            "batch_ms",
        )
    }
    errors: dict[str, int] = {}
    for i, body in enumerate(inputs):
        t0 = time.perf_counter()
        resp = client.post(url, json=body)
        e2e = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            errors[str(resp.status_code)] = errors.get(str(resp.status_code), 0) + 1
            continue
        if i < warmup:  # first requests pay for connection setup and cold caches
            continue
        t = parse_server_timing(resp.headers["Server-Timing"])
        samples["e2e_ms"].append(e2e)
        samples["inference_ms"].append(t["inference"])
        samples["overhead_ms"].append(e2e - t["inference"] - t.get("batch", 0.0))
        samples["http_ms"].append(e2e - t["app"])
        # batch: time in a forming batch (Stage 3+). At light load with one request in flight it
        # is the batch window (max_wait) unless the model doesn't batch; it's not overhead.
        samples["rpc_ms"].append(t["app"] - t["queue"] - t.get("batch", 0.0) - t["inference"])
        samples["batch_ms"].append(t.get("batch", 0.0))
        samples["queue_ms"].append(t["queue"])
    if not samples["e2e_ms"]:
        raise SystemExit(f"no successful requests (errors by status: {errors})")
    return {
        "measured_requests": len(samples["e2e_ms"]),
        "errors_by_status": errors,
        "stats_ms": {k: summarize(v) for k, v in samples.items()},
    }


def render(result: dict[str, Any]) -> str:
    rows = [f"{'':<14}{'p50':>9}{'p90':>9}{'p99':>9}{'max':>9}{'mean':>9}"]
    for name, s in result["stats_ms"].items():
        rows.append(
            f"{name:<14}" + "".join(f"{s[k]:>9.2f}" for k in ("p50", "p90", "p99", "max", "mean"))
        )
    errors = result["errors_by_status"] or "none"
    rows.append(f"\n{result['measured_requests']} requests measured; errors: {errors}")
    rows.append(LOWER_BOUND_NOTE)
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000", help="API base URL")
    p.add_argument("--model", choices=["sentiment", "iris"], default="iris")
    p.add_argument("--requests", type=int, default=1000, help="measured requests")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--words", type=int, default=46, help="sentiment: words per text (~48 tokens)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = p.parse_args()

    n = args.warmup + args.requests
    if args.model == "sentiment":
        inputs = make_texts(n, args.words, args.seed)
    else:
        inputs = make_iris_features(n, args.seed)

    # One keep-alive connection: the client reuses it, so connection setup isn't counted in
    # every request (as with any real client using a connection pool).
    with httpx.Client(base_url=args.url, timeout=5.0) as client:
        ready = client.get("/health/ready")
        if ready.status_code != 200:
            raise SystemExit(f"API not ready: {ready.status_code} {ready.text}")
        result = run(client, args.model, inputs, args.warmup)

    info = machine_info(threads=None)  # the worker's threads are set in the worker process
    prov = provenance("bench_overhead.py")
    print(f"{args.model}: sequential requests to {args.url}\n")
    print(render(result))

    out = args.out or Path("bench/results") / (
        f"stage2-overhead-{args.model}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "machine": info,
                "provenance": prov,
                "note": LOWER_BOUND_NOTE,
                "results": result,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    if prov["git"] is None or prov["git"]["dirty"]:
        print("warning: result not traceable to a commit (no git, or uncommitted changes)")


if __name__ == "__main__":
    main()
