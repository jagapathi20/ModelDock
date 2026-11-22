"""Measure how inference time scales with batch size.

    uv run python -m scripts.bench_batching --model-dir artifacts/sentiment-v1
    uv run python -m scripts.bench_batching --kind sklearn_classifier --model-dir artifacts/iris-v1

For each batch size, the script runs a few warm-up batches (not recorded),
then ``--iters`` timed batches, and reports per-batch time (p50/p90/p99),
per-item time, throughput, and the per-item speedup over batch size 1.
For the transformer it also splits time into tokenize / forward / postprocess.

It finishes with a latency budget check: plugging the measured batch times
into the D1/D5 worst-case formulas for the configured ``max_batch_size`` and
``max_wait``. Results (with hardware info) are written to a JSON file so the
numbers that end up in the README can be traced to a real run.

Only this process runs: no Redis, no HTTP. These are model-only numbers, a
lower bound on what the server can do.
"""

import argparse
import json
import os
import platform
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.provenance import provenance
from worker.runners import RUNNER_KINDS, ModelRunner, build_runner

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
    results: Sequence[BatchResult], max_batch_size: int, max_wait_ms: float, target_ms: float
) -> dict[str, Any] | None:
    """Worst-case request latency (model side only) under the D5 batching rule.

    With T = p99 time of a full batch (max_batch_size):

    - light load:  the request starts a batch, waits up to max_wait for it to
                   fill, then the batch runs:              max_wait + T
    - saturated:   a full batch is already running (compute thread busy); the
                   request's batch fills instantly, waits for it, then runs: 2T
    - at the backpressure threshold (D6's original limit of one full batch
                   queued per worker) the request also waits for that batch: 3T

    OUTDATED: this is the Stage 1 model, kept unchanged so the "latency_budget"
    block in existing result files can still be reproduced. It was corrected by
    C1.2 and then C1.11 (one worker loop, worst case T x (1 + ceil((d + 1) / B))),
    and the fixed queue limit was replaced by an explicit per-version
    max_queue_depth (C1.17). Stage 3 replaces this function. Until then, ignore
    its output and use the budgets in docs/DESIGN.md.

    Excludes Redis round trips, JSON, HTTP and the event loop; Stage 2 measures
    that overhead, and the Stage 7 load tests measure end-to-end latency.
    """
    full = next((r for r in results if r.batch_size == max_batch_size), None)
    if full is None:
        return None
    t = full.batch_ms_p99
    cases = {
        "light_load": max_wait_ms + t,
        "saturated": 2 * t,
        "at_backpressure_threshold": 3 * t,
    }
    return {
        "max_batch_size": max_batch_size,
        "max_wait_ms": max_wait_ms,
        "full_batch_ms_p99": t,
        "target_p99_ms": target_ms,
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
    try:
        import torch
        import transformers

        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        info["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


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
            f"Latency budget (model only), max_batch_size={budget['max_batch_size']}, "
            f"max_wait={budget['max_wait_ms']:g} ms, "
            f"full batch p99={budget['full_batch_ms_p99']:.1f} ms, "
            f"target p99 < {budget['target_p99_ms']:g} ms",
        ]
        for case, ms in budget["worst_case_ms"].items():
            ok = "ok" if budget["within_target"][case] else "OVER"
            lines.append(f"  {case:<27} {ms:8.1f} ms   {ok}")
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
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--max-batch-size", type=int, default=16, help="for the budget check")
    p.add_argument("--max-wait-ms", type=float, default=15.0, help="for the budget check")
    p.add_argument("--target-ms", type=float, default=150.0, help="p99 target (D1)")
    p.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = p.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    threads = None
    if args.kind == "transformer_classifier":
        from worker.runners.transformer import configure_cpu_threads

        threads = args.threads
        configure_cpu_threads(threads)

        def make_inputs(n: int, seed: int) -> list[dict[str, Any]]:
            return make_texts(n, args.words, seed)
    else:

        def make_inputs(n: int, seed: int) -> list[dict[str, Any]]:
            return make_iris_features(n, seed)

    runner = build_runner(args.kind)
    t0 = time.perf_counter()
    runner.load(args.model_dir)
    load_s = time.perf_counter() - t0

    info = machine_info(threads)
    prov = provenance("bench_batching.py")
    print(f"{args.kind} from {args.model_dir} (loaded in {load_s:.2f} s)")
    print(f"{info['cpu_model']}, {info['cpu_count_logical']} logical cores, threads={threads}")
    if args.kind == "transformer_classifier":
        print(f"{args.words} words per text")
    print()

    results = measure(runner, make_inputs, batch_sizes, args.warmup, args.iters)
    budget = latency_budget(results, args.max_batch_size, args.max_wait_ms, args.target_ms)
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
