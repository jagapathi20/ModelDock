"""Smoke tests for the benchmark script's logic (not for any performance numbers)."""

import pytest

pytest.importorskip("sklearn")

from scripts.bench_batching import (  # noqa: E402
    BatchResult,
    latency_budget,
    make_iris_features,
    make_texts,
    measure,
    percentile,
    render,
)
from scripts.train_iris import train  # noqa: E402
from worker.runners.tabular import SklearnClassifierRunner  # noqa: E402


def test_percentile_nearest_rank():
    values = list(range(1, 101))  # 1..100
    assert percentile(values, 50) == 50
    assert percentile(values, 99) == 99
    assert percentile([7.0], 99) == 7.0


def test_make_texts_is_deterministic_and_sized():
    a = make_texts(3, words_per_text=10, seed=1)
    assert a == make_texts(3, words_per_text=10, seed=1)
    assert a != make_texts(3, words_per_text=10, seed=2)
    assert all(len(x["text"].split()) == 10 for x in a)


def test_measure_and_render_end_to_end(tmp_path):
    train(tmp_path)
    runner = SklearnClassifierRunner()
    runner.load(tmp_path)

    results = measure(
        runner, lambda n, s: make_iris_features(n, s), batch_sizes=[1, 4], warmup=1, iters=3
    )
    assert [r.batch_size for r in results] == [1, 4]
    assert results[0].speedup_per_item_vs_bs1 == 1.0
    assert all(r.batch_ms_p50 > 0 for r in results)

    text = render(results, None)
    assert "ms/item" in text
    assert "p99 is close to the max" in text  # only 3 iters


def _result(bs: int, p99: float) -> BatchResult:
    return BatchResult(
        batch_size=bs,
        iters=100,
        batch_ms_p50=p99,
        batch_ms_p90=p99,
        batch_ms_p99=p99,
        batch_ms_mean=p99,
        item_ms_p50=p99 / bs,
        items_per_sec=bs / (p99 / 1000),
    )


def test_latency_budget_formulas():
    budget = latency_budget([_result(16, 40.0)], max_batch_size=16, max_wait_ms=15, target_ms=150)
    assert budget["worst_case_ms"] == {
        "light_load": 55.0,
        "saturated": 80.0,
        "at_backpressure_threshold": 120.0,
    }
    assert all(budget["within_target"].values())

    slow = latency_budget([_result(16, 60.0)], max_batch_size=16, max_wait_ms=15, target_ms=150)
    assert slow["within_target"]["at_backpressure_threshold"] is False


def test_latency_budget_needs_the_configured_batch_size():
    assert (
        latency_budget([_result(8, 10.0)], max_batch_size=16, max_wait_ms=15, target_ms=150) is None
    )


def test_server_timing_parsing():
    from scripts.bench_overhead import parse_server_timing

    header = "app;dur=2.500, queue;dur=0.250, inference;dur=1.000"
    assert parse_server_timing(header) == {"app": 2.5, "queue": 0.25, "inference": 1.0}
    assert parse_server_timing("cache;desc=hit") == {}  # entries without dur are skipped
