"""Measure how late Redis fires a BLPOP timeout when nothing arrives (evidence for C3.1).

    uv run python -m scripts.check_redis_timeouts

Redis checks blocked clients' timeouts on its internal timer (``hz``, 10 per second by default),
so a short BLPOP timeout can fire up to ~100 ms late. That's why the worker times the batch
window itself instead of using a BLPOP timeout. This script records the effect on the Redis you
actually run: for each timeout it does ``--samples`` BLPOPs on a key nothing pushes to, and
reports the elapsed time. It uses database 14 and a key no other code uses.
"""

import argparse
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import redis

from scripts.bench_batching import machine_info, percentile
from scripts.provenance import provenance
from scripts.redis_info import describe, redis_server_info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="redis://localhost:6379/14")
    p.add_argument("--timeouts-ms", default="1,5,15,50,200")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    r = redis.Redis.from_url(args.url, socket_timeout=5)
    server = redis_server_info(args.url)
    if server is None:
        raise SystemExit(f"no Redis at {args.url}: start it with `docker compose up -d`")
    print(describe(server) + "\n")
    key = f"timeout-probe:{uuid.uuid4().hex}"  # never pushed to

    results = []
    for timeout_ms in (float(x) for x in args.timeouts_ms.split(",")):
        elapsed = []
        for _ in range(args.samples):
            t0 = time.perf_counter()
            r.blpop([key], timeout=timeout_ms / 1000)
            elapsed.append((time.perf_counter() - t0) * 1000)
        row = {
            "timeout_ms": timeout_ms,
            "elapsed_ms_min": min(elapsed),
            "elapsed_ms_p50": percentile(elapsed, 50),
            "elapsed_ms_max": max(elapsed),
        }
        results.append(row)
        print(
            f"timeout {timeout_ms:6.0f} ms -> min {row['elapsed_ms_min']:6.1f}  "
            f"median {row['elapsed_ms_p50']:6.1f}  max {row['elapsed_ms_max']:6.1f} ms"
        )

    prov = provenance("check_redis_timeouts.py")
    out = args.out or Path("bench/results") / (
        f"stage3-redis-blpop-timeout-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "machine": machine_info(threads=None) | {"redis": server},
                "provenance": prov,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    if prov["git"] is None or prov["git"]["dirty"]:
        print("warning: result not traceable to a commit (no git, or uncommitted changes)")


if __name__ == "__main__":
    main()
