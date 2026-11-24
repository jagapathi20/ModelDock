"""The worker loop: batching (D5, C1.11, C3.1), deadlines (C2.3), failures (C3.3), replies."""

import threading
import time
import uuid

import fakeredis
import pytest
import redis

from common.models import IRIS_V1, ModelSpec
from common.protocol import REPLY_TTL_S, ReplyMessage, RequestMessage, now_ms, reply_key
from worker.loop import Worker, batch_window_s, check_spec_matches
from worker.runners import InvalidInputError, ModelRunner

BAD = -1.0  # FakeRunner rejects an input whose first feature is this


def batching_spec(max_batch_size: int = 4, max_wait_ms: float = 50.0) -> ModelSpec:
    """iris v1's queue and schema, but batching (the real iris v1 has batching off)."""
    return ModelSpec(
        name="iris",
        version=1,
        kind="sklearn_classifier",
        artifact="iris-v1",
        n_features=4,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )


class FakeRunner(ModelRunner):
    """Scores each input by its first feature, and records the size of every call.

    - an input whose first feature is ``BAD`` raises InvalidInputError naming its index;
    - ``fail_in_batch``: any call with more than one input raises RuntimeError;
    - ``fail_with``: every call raises this.
    """

    kind = "fake"
    n_features = 4

    def __init__(self, fail_with: Exception | None = None, fail_in_batch: bool = False) -> None:
        super().__init__()
        self._loaded = True
        self.fail_with = fail_with
        self.fail_in_batch = fail_in_batch
        self.calls: list[int] = []

    def load(self, model_dir) -> None:  # pragma: no cover - never called
        pass

    def predict_batch(self, inputs):
        self.calls.append(len(inputs))
        if self.fail_with is not None:
            raise self.fail_with
        if self.fail_in_batch and len(inputs) > 1:
            raise RuntimeError("only fails in a batch")
        for i, item in enumerate(inputs):
            if item["features"][0] == BAD:
                raise InvalidInputError(f"input {i}: bad", index=i)
        return [{"label": "x", "score": item["features"][0]} for item in inputs]


@pytest.fixture
def client(fake_server):
    return fakeredis.FakeRedis(server=fake_server)


def _message(first: float = 1.0, deadline_in_ms: float = 2000, **overrides) -> RequestMessage:
    fields = {
        "request_id": uuid.uuid4().hex,
        "reply_to": reply_key(uuid.uuid4().hex),
        "model": "iris",
        "version": 1,
        "input": {"features": [first, 2, 3, 4]},
        "deadline_ms": now_ms() + deadline_in_ms,
        "enqueued_at_ms": now_ms(),
    }
    return RequestMessage(**(fields | overrides))


def _enqueue(client, *messages: RequestMessage) -> list[RequestMessage]:
    for m in messages:
        client.rpush(IRIS_V1.queue, m.encode())
    return list(messages)


def _reply(client, request: RequestMessage) -> ReplyMessage | None:
    raw = client.lpop(request.reply_to)
    return None if raw is None else ReplyMessage.decode(raw)


# --- one request at a time (iris v1: batching off) ---------------------------------------------


def test_request_gets_a_reply_with_expiry(client):
    worker = Worker(IRIS_V1, FakeRunner(), client)
    [request] = _enqueue(client, _message())

    assert worker.run_once(block_timeout_s=0.1) == 1
    assert 0 < client.ttl(request.reply_to) <= REPLY_TTL_S  # set by the worker (C1.18)

    reply = _reply(client, request)
    assert reply.request_id == request.request_id
    assert reply.status == "ok"
    assert reply.output == {"label": "x", "score": 1.0}
    assert reply.timings.batch_size == 1
    assert reply.timings.inference_ms >= 0 and reply.timings.batch_wait_ms >= 0
    assert worker.stats.processed == 1


def test_batching_off_never_waits(client):
    worker = Worker(IRIS_V1, FakeRunner(), client)  # max_batch_size 1
    _enqueue(client, _message(), _message())
    t0 = time.perf_counter()
    worker.run_once(0.1)
    assert time.perf_counter() - t0 < 0.05
    assert worker.runner.calls == [1]


def test_idle_iteration_returns_zero_and_updates_heartbeat(client):
    worker = Worker(IRIS_V1, FakeRunner(), client)
    before = worker.last_loop_at
    assert worker.run_once(block_timeout_s=0.05) == 0
    assert worker.last_loop_at > before


# --- batching: size and time triggers -----------------------------------------------------------


def test_full_batch_runs_at_once(client):
    """Size trigger: four requests already queued form one batch without waiting."""
    runner = FakeRunner()
    worker = Worker(batching_spec(max_wait_ms=1000), runner, client)
    requests = _enqueue(client, *(_message(first=float(i)) for i in range(4)))

    t0 = time.perf_counter()
    assert worker.run_once(0.1) == 4
    assert time.perf_counter() - t0 < 0.2  # well under the 1 s max_wait
    assert runner.calls == [4]
    for i, r in enumerate(requests):  # each reply is its own request's (order kept)
        reply = _reply(client, r)
        assert reply.output["score"] == float(i)
        assert reply.timings.batch_size == 4


def test_lone_request_runs_after_max_wait(client):
    """Time trigger: a single request waits max_wait for company, then runs alone."""
    runner = FakeRunner()
    worker = Worker(batching_spec(max_wait_ms=60), runner, client)
    [request] = _enqueue(client, _message())

    t0 = time.perf_counter()
    worker.run_once(0.1)
    elapsed = time.perf_counter() - t0
    assert 0.05 <= elapsed < 0.15
    assert runner.calls == [1]
    assert _reply(client, request).timings.batch_wait_ms >= 50


def test_late_arrival_joins_the_open_batch(client, fake_server):
    runner = FakeRunner()
    worker = Worker(batching_spec(max_wait_ms=300), runner, client)
    first, late = _message(first=1.0), _message(first=2.0)
    _enqueue(client, first)
    other = fakeredis.FakeRedis(server=fake_server)
    timer = threading.Timer(0.03, lambda: other.rpush(IRIS_V1.queue, late.encode()))
    timer.start()

    worker.run_once(0.5)
    timer.join()
    assert runner.calls == [2]
    assert _reply(client, late).timings.batch_size == 2


def test_batch_closes_soon_after_it_fills(client, fake_server):
    """Polling closes a batch within a few ms of its last request, not at max_wait."""
    worker = Worker(batching_spec(max_batch_size=2, max_wait_ms=1000), FakeRunner(), client)
    _enqueue(client, _message())
    other = fakeredis.FakeRedis(server=fake_server)
    threading.Timer(0.02, lambda: other.rpush(IRIS_V1.queue, _message().encode())).start()

    t0 = time.perf_counter()
    worker.run_once(0.5)
    assert time.perf_counter() - t0 < 0.2  # far from the 1 s max_wait


def test_more_than_a_batch_waiting_splits_fifo(client):
    runner = FakeRunner()
    worker = Worker(batching_spec(max_wait_ms=10), runner, client)
    requests = _enqueue(client, *(_message(first=float(i)) for i in range(6)))
    worker.run_once(0.1)
    worker.run_once(0.1)
    assert runner.calls == [4, 2]
    assert [_reply(client, r).timings.batch_size for r in requests] == [4, 4, 4, 4, 2, 2]


# --- the batch window and clock skew (C3.1) -------------------------------------------------------


@pytest.mark.parametrize(
    ("queued_ms", "expected_s"),
    [
        (0, 0.015),  # just arrived: the whole window
        (5, 0.010),  # already queued 5 ms: 10 ms left
        (40, 0.0),  # queued longer than max_wait: run at once
        (-10_000, 0.015),  # worker clock behind (negative queue time): never more than max_wait
    ],
)
def test_batch_window(queued_ms, expected_s):
    now = 1_000_000.0
    assert batch_window_s(15, now - queued_ms, now) == pytest.approx(expected_s)


def test_worker_clock_ahead_closes_the_batch_at_once(client):
    worker = Worker(batching_spec(max_wait_ms=500), FakeRunner(), client)
    _enqueue(client, _message(enqueued_at_ms=now_ms() - 10_000))  # looks queued for 10 s
    t0 = time.perf_counter()
    worker.run_once(0.1)
    assert time.perf_counter() - t0 < 0.1


def test_worker_clock_behind_waits_at_most_max_wait(client):
    worker = Worker(batching_spec(max_wait_ms=60), FakeRunner(), client)
    _enqueue(client, _message(enqueued_at_ms=now_ms() + 10_000))  # "enqueued" 10 s from now
    t0 = time.perf_counter()
    worker.run_once(0.1)
    assert time.perf_counter() - t0 < 0.2  # ~60 ms, not 10 s


# --- deadlines (C2.3) ---------------------------------------------------------------------------


def test_expired_at_pop_is_dropped_without_reply_or_compute(client):
    runner = FakeRunner()
    worker = Worker(IRIS_V1, runner, client)
    [request] = _enqueue(client, _message(deadline_in_ms=-1))

    assert worker.run_once(0.1) == 1
    assert runner.calls == []  # no CPU spent on a client that already got 504
    assert _reply(client, request) is None
    assert worker.stats.expired == 1


def test_expiring_while_the_batch_forms_is_dropped_before_compute(client):
    """The second check: alive at pop, past its deadline by the time the batch runs."""
    runner = FakeRunner()
    worker = Worker(batching_spec(max_wait_ms=100), runner, client)
    short, long = _enqueue(client, _message(deadline_in_ms=30), _message(first=2.0))

    worker.run_once(0.1)
    assert runner.calls == [1]  # only the one still alive
    assert _reply(client, short) is None
    assert _reply(client, long).status == "ok"
    assert worker.stats.expired == 1


# --- failures (C3.3) --------------------------------------------------------------------------


def test_schema_invalid_input_is_answered_without_the_model(client):
    runner = FakeRunner()
    worker = Worker(batching_spec(max_wait_ms=10), runner, client)
    good = _message()
    bad = _message(input={"features": [1, 2, 3]})  # 3 features: fails the catalog schema
    _enqueue(client, good, bad)

    worker.run_once(0.1)
    assert runner.calls == [1]  # the bad input never reached the model
    assert _reply(client, good).status == "ok"
    bad_reply = _reply(client, bad)
    assert bad_reply.error.code == "invalid_input"
    assert "features" in bad_reply.error.message


def test_runner_rejecting_one_input_fails_only_that_request(client):
    """InvalidInputError with an index: 422 for that one; the rest run again as one batch."""
    runner = FakeRunner()
    worker = Worker(batching_spec(max_wait_ms=10), runner, client)
    a, bad, b = _enqueue(client, _message(1.0), _message(BAD), _message(3.0))

    worker.run_once(0.1)
    assert runner.calls == [3, 2]  # one extra batch, not three solo runs
    assert _reply(client, bad).error.code == "invalid_input"
    for r, score in ((a, 1.0), (b, 3.0)):
        reply = _reply(client, r)
        assert reply.output["score"] == score
        assert reply.timings.batch_size == 2


def test_failed_batch_is_rerun_one_request_at_a_time(client):
    runner = FakeRunner(fail_in_batch=True)
    worker = Worker(batching_spec(max_wait_ms=10), runner, client)
    requests = _enqueue(client, *(_message(float(i)) for i in range(3)))

    worker.run_once(0.1)
    assert runner.calls == [3, 1, 1, 1]
    assert worker.stats.rerun_alone == 1
    for i, r in enumerate(requests):
        reply = _reply(client, r)
        assert reply.status == "ok" and reply.output["score"] == float(i)
        assert reply.timings.batch_size == 1


def test_requests_that_fail_alone_too_get_500(client):
    runner = FakeRunner(fail_with=RuntimeError("model broken"))
    worker = Worker(batching_spec(max_wait_ms=10), runner, client)
    requests = _enqueue(client, _message(), _message())

    worker.run_once(0.1)
    for r in requests:
        reply = _reply(client, r)
        assert reply.error.code == "inference_failed"
        assert reply.error.message == "inference failed"  # internals stay in the worker's log


@pytest.mark.parametrize("exc", [ValueError("numpy said no"), KeyError("x")])
def test_plain_exceptions_are_inference_failed_not_invalid_input(client, exc):
    # A plain ValueError from inside numpy/torch is our failure, not the client's (C2.1).
    worker = Worker(IRIS_V1, FakeRunner(exc), client)
    [request] = _enqueue(client, _message())
    worker.run_once(0.1)
    assert _reply(client, request).error.code == "inference_failed"


def test_invalid_input_error_without_index_falls_back_to_solo_runs(client):
    class NoIndex(FakeRunner):
        def predict_batch(self, inputs):
            if len(inputs) > 1:
                self.calls.append(len(inputs))
                raise InvalidInputError("something in this batch is bad")  # no index
            return super().predict_batch(inputs)

    runner = NoIndex()
    worker = Worker(batching_spec(max_wait_ms=10), runner, client)
    ok, bad = _enqueue(client, _message(1.0), _message(BAD))
    worker.run_once(0.1)
    assert _reply(client, ok).status == "ok"
    assert _reply(client, bad).error.code == "invalid_input"


def test_worker_keeps_running_after_a_failure(client):
    runner = FakeRunner(RuntimeError("boom"))
    worker = Worker(IRIS_V1, runner, client)
    _enqueue(client, _message())
    worker.run_once(0.1)

    runner.fail_with = None
    [request] = _enqueue(client, _message())
    worker.run_once(0.1)
    assert _reply(client, request).status == "ok"


@pytest.mark.parametrize(
    "raw",
    [b"not json", b"{}", b'{"request_id": "x"}', b'{"reply_to": "q:sentiment:1"}'],
)
def test_malformed_message_is_dropped(client, raw):
    worker = Worker(IRIS_V1, FakeRunner(), client)
    client.rpush(IRIS_V1.queue, raw)
    assert worker.run_once(0.1) == 1
    assert worker.stats.malformed == 1
    assert client.keys("replies:*") == []  # nothing pushed anywhere


def test_malformed_first_message_does_not_block_the_batch(client):
    runner = FakeRunner()
    worker = Worker(batching_spec(max_wait_ms=10), runner, client)
    client.rpush(IRIS_V1.queue, b"garbage")
    requests = _enqueue(client, _message(), _message())
    worker.run_once(0.1)
    assert runner.calls == [2]
    assert all(_reply(client, r).status == "ok" for r in requests)


def test_run_survives_redis_errors(client, monkeypatch):
    """run() backs off on a Redis error and continues; it doesn't exit."""
    worker = Worker(IRIS_V1, FakeRunner(), client)
    [request] = _enqueue(client, _message())
    calls = {"n": 0}
    real_run_once = worker.run_once

    def flaky_run_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise redis.ConnectionError("Redis restarting")
        result = real_run_once(0.1)
        worker.stop()
        return result

    monkeypatch.setattr(worker, "run_once", flaky_run_once)
    monkeypatch.setattr("worker.loop.time.sleep", lambda s: None)
    worker.run()
    assert worker.stats.redis_errors == 2
    assert _reply(client, request).status == "ok"


def test_spec_drift_check():
    check_spec_matches(IRIS_V1, FakeRunner())  # 4 == 4: fine

    runner = FakeRunner()
    runner.n_features = 5
    with pytest.raises(RuntimeError, match="catalog says 4 features"):
        check_spec_matches(IRIS_V1, runner)
