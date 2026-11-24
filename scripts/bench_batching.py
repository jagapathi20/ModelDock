"""Measure how inference time scales with batch size (model only: no Redis, no HTTP).

    uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1 --threads 4 \
        --max-length 48 --words 46
    uv run python -m scripts.bench_batching --kind sklearn_classifier --model-dir artifacts/iris-v1

For each batch size, the script runs a few warm-up batches (not recorded), then ``--iters``
timed batches, and reports per-batch time (p50/p90/p99), per-item time, throughput, and the
per-item speedup over batch size 1. For the transformer it also splits time into tokenize /
forward / postprocess. On macOS it records the chip and its performance/efficiency core split,
so a thread-count experiment on Apple Silicon can be interpreted.

It finishes with the **latency budget** (C1.11) for ``--max-batch-size`` and ``--max-wait-ms``,
using T = the measured p99 of a full batch and ``--overhead-ms`` for everything outside
inference, and prints the **recommended ``max_queue_depth``** (C1.17), which is set by hand.

**History (C3.4).** Stage 1 had two copies of this script: ``bench_batching.py`` (run A, S) and
``bench_batching_1.py`` (runs B to K), identical except that the copy added the P/E core report.
Stage 3 merged them into this file. To reproduce a Stage 1 run exactly, check out the Stage 1.5
commit, where both copies and the Stage 1 budget formulas still exist:
``git log --diff-filter=D --oneline -- scripts/bench_batching_1.py`` names the commit that
removed the copy; its parent is the one to check out.

**Defaults that preserve Stage 1 commands.** ``--max-length`` defaults to 128 and ``--threads``
to all logical cores, as in Stage 1, so the commands recorded for runs A to K still mean what
they meant. Both are written to each result file's ``args``. New runs pass ``--max-length 48``
(C1.13) and ``--threads 4`` (C1.5).
"""

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.provenance import provenance
from worker.runners import RUNNER_KINDS, ModelRunner, build_runner
from worker.threads import cap_native_threads

# Neutral words plus a few sentiment-bearing ones. Random sentences are
# meaningless, but the model's cost depends only on token count, not meaning.
_WORDS = (
    "the a this that movie film plot actor scene story ending music camera script "
    "was is felt seemed really quite very not too somewhat rather "
    "good great bad boring brilliant awful slow fun moving dull clever weak strong "
    "and but so because although while with without about for in on at of"
).split()


def make_texts(n: int, words_per_text: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [{"text": " ".join(rng.choices(_WORDS, k=words_per_text))} for _ in range(n)]


def make_iris_features(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ranges = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]  # iris feature ranges
    return [{"features": [rng.uniform(lo, hi) for lo, hi in ranges]} for _ in range(n)]


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile. With n samples, p99 is meaningful only if n >= 100."""
    ordered = sorted(values)
    rank = max(1, round(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


@dataclass
class BatchResult:
    batch_size: int
    iters: int
    batch_ms_p50: float
    batch_ms_p90: float
    batch_ms_p99: float
    batch_ms_mean: float
    item_ms_p50: float
    items_per_sec: float
    speedup_per_item_vs_bs1: float = 1.0
    # Median ms per phase (transformer only), and mean padded sequence length.
    phases_ms_p50: dict[str, float] = field(default_factory=dict)
    padded_tokens_mean: float | None = None


def _time_one_batch(runner: ModelRunner, batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Run one batch and return its timings in ms (plus padded length if known)."""
    if all(hasattr(runner, a) for a in ("preprocess", "forward", "postprocess")):
        t0 = time.perf_counter_ns()
        encoded = runner.preprocess(batch)  # type: ignore[attr-defined]
        t1 = time.perf_counter_ns()
        logits = runner.forward(encoded)  # type: ignore[attr-defined]
        t2 = time.perf_counter_ns()
        runner.postprocess(logits)  # type: ignore[attr-defined]
        t3 = time.perf_counter_ns()
        return {
            "total": (t3 - t0) / 1e6,
            "tokenize": (t1 - t0) / 1e6,
            "forward": (t2 - t1) / 1e6,
            "postprocess": (t3 - t2) / 1e6,
            "padded_tokens": int(encoded["input_ids"].shape[1]),
        }
    t0 = time.perf_counter_ns()
    runner.predict_batch(batch)
    return {"total": (time.perf_counter_ns() - t0) / 1e6}


def measure(
    runner: ModelRunner,
    make_inputs: Callable[[int, int], list[dict[str, Any]]],
    batch_sizes: Sequence[int],
    warmup: int,
    iters: int,
) -> list[BatchResult]:
    """Time ``iters`` batches at each batch size. Each batch gets fresh inputs."""
    results: list[BatchResult] = []
    seed = 0
    for bs in batch_sizes:
        for _ in range(warmup):
            seed += 1
            _time_one_batch(runner, make_inputs(bs, seed))
        samples = []
        for _ in range(iters):
            seed += 1
            samples.append(_time_one_batch(runner, make_inputs(bs, seed)))

        totals = [s["total"] for s in samples]
        p50 = statistics.median(totals)
        phases = {
            k: statistics.median(s[k] for s in samples)
            for k in ("tokenize", "forward", "postprocess")
            if k in samples[0]
        }
        padded = (
            statistics.mean(s["padded_tokens"] for s in samples)
            if "padded_tokens" in samples[0]
            else None
        )
        results.append(
            BatchResult(
                batch_size=bs,
                iters=iters,
                batch_ms_p50=p50,
                batch_ms_p90=percentile(totals, 90),
                batch_ms_p99=percentile(totals, 99),
                batch_ms_mean=statistics.fmean(totals),
                item_ms_p50=p50 / bs,
                items_per_sec=bs / (p50 / 1000) if p50 > 0 else float("inf"),
                phases_ms_p50=phases,
                padded_tokens_mean=padded,
            )
        )

    base = next((r for r in results if r.batch_size == 1), None)
    if base:
        for r in results:
            r.speedup_per_item_vs_bs1 = base.item_ms_p50 / r.item_ms_p50 if r.item_ms_p50 else 0
    return results


def latency_budget(
    results: Sequence[BatchResult],
    max_batch_size: int,
    max_wait_ms: float,
    target_ms: float,
    overhead_ms: float,
    workers: int = 1,
) -> dict[str, Any] | None:
    """Worst-case latency of an accepted request, and the queue limit that keeps it on target.

    The worker runs one loop and takes a batch only when compute is free (C1.11). With
    T = p99 time of a full batch (``max_batch_size`` = B) and d requests ahead in Redis:

    - light load:          the request starts a batch, which waits up to max_wait to fill:
                           max_wait + T + overhead
    - saturated, empty queue: a full batch is running; this request's batch runs next:
                           2T + overhead
    - d requests ahead:    T x (1 + ceil((d + 1) / B)) + overhead

    The queue limit is the largest that keeps the last accepted request within the target:

        m = floor((target - overhead) / T) - 1          # batches from the queue that fit
        max_queue_depth = m x B x workers                # 503 when LLEN >= this (C1.17)

    At the limit an accepted request has at most max_queue_depth - 1 requests ahead. With
    m <= 0 even an empty queue misses the target, and no limit is recommended.

    ``overhead_ms`` is everything outside inference (HTTP, Redis, JSON). C2.6 measured it at light
    load, which makes it a lower bound: the recommendation is optimistic until a load test
    checks it (P10).
    """
    full = next((r for r in results if r.batch_size == max_batch_size), None)
    if full is None:
        return None
    t = full.batch_ms_p99
    b = max_batch_size
    m = math.floor((target_ms - overhead_ms) / t) - 1
    per_worker = max(m, 0) * b
    depth = per_worker * workers
    cases = {
        "light_load": max_wait_ms + t + overhead_ms,
        "saturated_empty_queue": 2 * t + overhead_ms,
    }
    if per_worker > 0:
        ahead = per_worker - 1  # per worker: each worker drains its share of the queue
        cases["at_queue_limit"] = t * (1 + math.ceil((ahead + 1) / b)) + overhead_ms
    return {
        "max_batch_size": b,
        "max_wait_ms": max_wait_ms,
        "full_batch_ms_p99": t,
        "overhead_ms": overhead_ms,
        "target_p99_ms": target_ms,
        "workers": workers,
        "batches_from_queue_m": m,
        "recommended_max_queue_depth": depth if depth > 0 else None,
        "worst_case_ms": cases,
        "within_target": {k: v < target_ms for k, v in cases.items()},
    }


def machine_info(threads: int | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "inference_threads": threads,
    }
    # Apple Silicon mixes performance (P) and efficiency (E) cores. Thread
    # counts above the P-core count put work on slower E-cores, which shows
    # up as tail latency, so record the split.
    p_cores, e_cores = _sysctl("hw.perflevel0.physicalcpu"), _sysctl("hw.perflevel1.physicalcpu")
    if p_cores:
        info["performance_cores"] = int(p_cores)
    if e_cores:
        info["efficiency_cores"] = int(e_cores)
    try:
        import torch
        import transformers

        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        info["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def _sysctl(name: str) -> str | None:
    """Read a macOS sysctl value; None on other platforms or if missing."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # macOS: e.g. "Apple M2". platform.processor() only says "arm".
    return _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"


def render(results: Sequence[BatchResult], budget: dict[str, Any] | None) -> str:
    has_phases = bool(results and results[0].phases_ms_p50)
    head = (
        f"{'batch':>5} {'p50 ms':>9} {'p90 ms':>9} {'p99 ms':>9} "
        f"{'ms/item':>9} {'items/s':>9} {'speedup':>8}"
    )
    if has_phases:
        head += f" {'tok ms':>8} {'fwd ms':>8} {'post ms':>8} {'tokens':>7}"
    lines = [head, "-" * len(head)]
    for r in results:
        line = (
            f"{r.batch_size:>5} {r.batch_ms_p50:>9.2f} {r.batch_ms_p90:>9.2f} "
            f"{r.batch_ms_p99:>9.2f} {r.item_ms_p50:>9.3f} {r.items_per_sec:>9.0f} "
            f"{r.speedup_per_item_vs_bs1:>7.1f}x"
        )
        if has_phases:
            ph = r.phases_ms_p50
            line += (
                f" {ph['tokenize']:>8.2f} {ph['forward']:>8.2f} {ph['postprocess']:>8.2f}"
                f" {r.padded_tokens_mean:>7.1f}"
            )
        lines.append(line)
    if results and results[0].iters < 100:
        lines.append(f"(note: {results[0].iters} iters per size; p99 is close to the max)")

    if budget:
        lines += [
            "",
            f"Latency budget (C1.11): max_batch_size={budget['max_batch_size']}, "
            f"max_wait={budget['max_wait_ms']:g} ms, T (full batch p99)="
            f"{budget['full_batch_ms_p99']:.1f} ms, overhead={budget['overhead_ms']:g} ms, "
            f"target p99 < {budget['target_p99_ms']:g} ms",
        ]
        for case, ms in budget["worst_case_ms"].items():
            ok = "ok" if budget["within_target"][case] else "OVER"
            lines.append(f"  {case:<27} {ms:8.1f} ms   {ok}")
        depth = budget["recommended_max_queue_depth"]
        if depth:
            lines.append(
                f"  recommended max_queue_depth: {depth} "
                f"(m={budget['batches_from_queue_m']}, workers={budget['workers']})"
            )
        else:
            lines.append("  no queue limit meets the target: even an empty queue misses it")
        lines.append(
            "  (overhead measured at light load is a lower bound, so this is optimistic; C2.6)"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--kind", choices=RUNNER_KINDS, default="transformer_classifier")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    p.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count(),
        help="PyTorch intra-op threads (default: all logical cores)",
    )
    p.add_argument("--words", type=int, default=24, help="words per input text")
    p.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="transformer: truncate inputs to this many tokens. Default 128 as in Stage 1; "
        "the catalog's sentiment v1 uses 48 (C1.13)",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--max-batch-size", type=int, default=4, help="for the budget (C1.14)")
    p.add_argument("--max-wait-ms", type=float, default=15.0, help="for the budget (D5)")
    p.add_argument("--target-ms", type=float, default=150.0, help="p99 target (C1.12)")
    p.add_argument(
        "--overhead-ms",
        type=float,
        # Sentiment p99 overhead at light load, C2.6:
        # bench/results/stage2-overhead-sentiment-20261122T093415Z.json
        default=1.77,
        help="everything outside inference, for the budget (default: C2.6's measurement)",
    )
    p.add_argument("--workers", type=int, default=1, help="workers for this version (C1.17)")
    p.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = p.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    threads = None
    if args.kind == "transformer_classifier":
        # Before numpy is first imported: BLAS pools read their limit once, at import (C3.2).
        cap_native_threads(args.threads)
        from worker.runners.transformer import configure_cpu_threads

        threads = args.threads
        configure_cpu_threads(threads)

        def make_inputs(n: int, seed: int) -> list[dict[str, Any]]:
            return make_texts(n, args.words, seed)
    else:

        def make_inputs(n: int, seed: int) -> list[dict[str, Any]]:
            return make_iris_features(n, seed)

    options = {"max_length": args.max_length} if args.kind == "transformer_classifier" else {}
    runner = build_runner(args.kind, **options)
    t0 = time.perf_counter()
    runner.load(args.model_dir)
    load_s = time.perf_counter() - t0

    info = machine_info(threads)
    prov = provenance("bench_batching.py")
    print(f"{args.kind} from {args.model_dir} (loaded in {load_s:.2f} s)")
    cores = f"{info['cpu_count_logical']} logical cores"
    if "performance_cores" in info:
        cores += f" ({info['performance_cores']}P + {info.get('efficiency_cores', 0)}E)"
    print(f"{info['cpu_model']}, {cores}, threads={threads}")
    if args.kind == "transformer_classifier":
        print(f"{args.words} words per text")
    print()

    results = measure(runner, make_inputs, batch_sizes, args.warmup, args.iters)
    budget = latency_budget(
        results,
        args.max_batch_size,
        args.max_wait_ms,
        args.target_ms,
        overhead_ms=args.overhead_ms,
        workers=args.workers,
    )
    print(render(results, budget))

    out = args.out or Path("bench/results") / (
        f"stage1-{args.kind}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "machine": info,
                "provenance": prov,
                "load_seconds": load_s,
                "results": [asdict(r) for r in results],
                "latency_budget": budget,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    git = prov["git"]
    if git is None:
        print("warning: git commit not recorded (no git, or not a repository)")
    elif git["dirty"]:
        print("warning: uncommitted changes; this result is not traceable to a commit")


if __name__ == "__main__":
    main()
