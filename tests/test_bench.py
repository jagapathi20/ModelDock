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


def test_latency_budget_c111_formulas():
    """T = 40, B = 4, overhead = 2, target 150 (C1.11).

    light = 15 + 40 + 2 = 57; saturated, empty queue = 2T + 2 = 82;
    m = floor((150 - 2) / 40) - 1 = 2, so max_queue_depth = 2 x 4 = 8 per worker;
    at the limit 7 are ahead: T x (1 + ceil(8 / 4)) + 2 = 122.
    """
    budget = latency_budget(
        [_result(4, 40.0)], max_batch_size=4, max_wait_ms=15, target_ms=150, overhead_ms=2
    )
    assert budget["worst_case_ms"] == {
        "light_load": 57.0,
        "saturated_empty_queue": 82.0,
        "at_queue_limit": 122.0,
    }
    assert budget["batches_from_queue_m"] == 2
    assert budget["recommended_max_queue_depth"] == 8
    assert all(budget["within_target"].values())


def test_queue_limit_scales_with_workers():
    budget = latency_budget(
        [_result(4, 40.0)],
        max_batch_size=4,
        max_wait_ms=15,
        target_ms=150,
        overhead_ms=2,
        workers=2,
    )
    assert budget["recommended_max_queue_depth"] == 16  # total for the version's queue (C1.17)
    assert budget["worst_case_ms"]["at_queue_limit"] == 122.0  # per worker, unchanged


def test_no_queue_limit_when_even_an_empty_queue_misses():
    # T = 80: 2T + 2 = 162 > 150, and m = floor(148 / 80) - 1 = 0.
    budget = latency_budget(
        [_result(4, 80.0)], max_batch_size=4, max_wait_ms=15, target_ms=150, overhead_ms=2
    )
    assert budget["recommended_max_queue_depth"] is None
    assert "at_queue_limit" not in budget["worst_case_ms"]
    assert budget["within_target"]["saturated_empty_queue"] is False
    assert "no queue limit meets the target" in render([_result(4, 80.0)], budget)


def test_stage2_numbers_reproduce_c26():
    """The budget function gives C2.6's figures: run I's T at batch 4 with 9.00 ms overhead
    (Compose Redis re-measurement), and with the superseded 1.77 ms."""
    budget = latency_budget(
        [_result(4, 38.9)], max_batch_size=4, max_wait_ms=15, target_ms=150, overhead_ms=9.0
    )
    assert budget["recommended_max_queue_depth"] == 8
    assert budget["worst_case_ms"]["at_queue_limit"] == pytest.approx(125.7)
    assert budget["worst_case_ms"]["saturated_empty_queue"] == pytest.approx(86.8)
    old = latency_budget(
        [_result(4, 38.9)], max_batch_size=4, max_wait_ms=15, target_ms=150, overhead_ms=1.77
    )
    assert old["worst_case_ms"]["at_queue_limit"] == pytest.approx(118.47)


def test_latency_budget_needs_the_configured_batch_size():
    assert (
        latency_budget(
            [_result(8, 10.0)], max_batch_size=16, max_wait_ms=15, target_ms=150, overhead_ms=2
        )
        is None
    )


def test_server_timing_parsing():
    from scripts.bench_overhead import parse_server_timing

    header = "app;dur=2.500, queue;dur=0.250, inference;dur=1.000"
    assert parse_server_timing(header) == {"app": 2.5, "queue": 0.25, "inference": 1.0}
    assert parse_server_timing("cache;desc=hit") == {}  # entries without dur are skipped
