"""The worker loop (Stage 2): pop, deadline check, infer, reply with RPUSH + EXPIRE."""

import uuid

import fakeredis
import pytest
import redis

from common.models import IRIS_V1
from common.protocol import REPLY_TTL_S, ReplyMessage, RequestMessage, now_ms, reply_key
from worker.loop import Worker, check_spec_matches
from worker.runners import InvalidInputError, ModelRunner


class FakeRunner(ModelRunner):
    """Echoes the feature count; ``fail_with`` makes predict_batch raise instead."""

    kind = "fake"
    n_features = 4

    def __init__(self, fail_with: Exception | None = None) -> None:
        super().__init__()
        self._loaded = True
        self.fail_with = fail_with
        self.calls = 0

    def load(self, model_dir) -> None:  # pragma: no cover - never called
        pass

    def predict_batch(self, inputs):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return [{"label": "x", "score": float(len(i["features"]))} for i in inputs]


@pytest.fixture
def client(fake_server):
    return fakeredis.FakeRedis(server=fake_server)


def _enqueue(client, deadline_in_ms: int = 500, **overrides) -> RequestMessage:
    msg = RequestMessage(
        **{
            "request_id": uuid.uuid4().hex,
            "reply_to": reply_key(uuid.uuid4().hex),
            "model": "iris",
            "version": 1,
            "input": {"features": [1, 2, 3, 4]},
            "deadline_ms": now_ms() + deadline_in_ms,
            "enqueued_at_ms": now_ms(),
        }
        | overrides
    )
    client.rpush(IRIS_V1.queue, msg.encode())
    return msg


def _reply(client, request: RequestMessage) -> ReplyMessage | None:
    raw = client.lpop(request.reply_to)
    return None if raw is None else ReplyMessage.decode(raw)


def test_request_gets_a_reply_with_expiry(client):
    worker = Worker(IRIS_V1, FakeRunner(), client)
    request = _enqueue(client)

    assert worker.run_once(block_timeout_s=0.1) is True
    # The worker sets the expiry when it pushes (C1.18).
    assert 0 < client.ttl(request.reply_to) <= REPLY_TTL_S

    reply = _reply(client, request)
    assert reply.request_id == request.request_id
    assert reply.status == "ok"
    assert reply.output == {"label": "x", "score": 4.0}
    assert reply.timings.batch_size == 1
    assert reply.timings.inference_ms >= 0
    assert worker.stats.processed == 1


def test_idle_iteration_returns_false_and_updates_heartbeat(client):
    worker = Worker(IRIS_V1, FakeRunner(), client)
    before = worker.last_loop_at
    assert worker.run_once(block_timeout_s=0.05) is False
    assert worker.last_loop_at > before


def test_requests_are_served_in_arrival_order(client):
    worker = Worker(IRIS_V1, FakeRunner(), client)
    first, second = _enqueue(client), _enqueue(client)
    worker.run_once(0.1)
    assert _reply(client, first) is not None
    assert _reply(client, second) is None
    worker.run_once(0.1)
    assert _reply(client, second) is not None


def test_expired_request_is_dropped_without_reply_or_compute(client):
    runner = FakeRunner()
    worker = Worker(IRIS_V1, runner, client)
    request = _enqueue(client, deadline_in_ms=-1)

    assert worker.run_once(0.1) is True
    assert runner.calls == 0  # no CPU spent on a client that already got 504
    assert _reply(client, request) is None
    assert worker.stats.expired == 1


def test_invalid_input_error_becomes_invalid_input_reply(client):
    worker = Worker(IRIS_V1, FakeRunner(InvalidInputError("input 0: bad", index=0)), client)
    request = _enqueue(client)
    worker.run_once(0.1)

    reply = _reply(client, request)
    assert reply.status == "error"
    assert reply.error.code == "invalid_input"
    assert reply.error.message == "input 0: bad"
    assert worker.stats.invalid_input == 1


@pytest.mark.parametrize("exc", [ValueError("numpy said no"), RuntimeError("boom"), KeyError("x")])
def test_other_exceptions_become_inference_failed(client, exc):
    # A plain ValueError from inside numpy/torch is our failure, not the client's (C2.1).
    worker = Worker(IRIS_V1, FakeRunner(exc), client)
    request = _enqueue(client)
    worker.run_once(0.1)

    reply = _reply(client, request)
    assert reply.error.code == "inference_failed"
    assert reply.error.message == "inference failed"  # internals stay in the worker's log
    assert worker.stats.inference_failed == 1


def test_worker_keeps_running_after_a_failure(client):
    runner = FakeRunner(RuntimeError("boom"))
    worker = Worker(IRIS_V1, runner, client)
    _enqueue(client)
    worker.run_once(0.1)

    runner.fail_with = None
    request = _enqueue(client)
    worker.run_once(0.1)
    assert _reply(client, request).status == "ok"


@pytest.mark.parametrize(
    "raw",
    [b"not json", b"{}", b'{"request_id": "x"}', b'{"reply_to": "q:sentiment:1"}'],
)
def test_malformed_message_is_dropped(client, raw):
    worker = Worker(IRIS_V1, FakeRunner(), client)
    client.rpush(IRIS_V1.queue, raw)
    assert worker.run_once(0.1) is True
    assert worker.stats.malformed == 1
    assert client.keys("replies:*") == []  # nothing pushed anywhere


def test_run_survives_redis_errors(client, monkeypatch):
    """run() backs off on a Redis error and continues; it doesn't exit."""
    worker = Worker(IRIS_V1, FakeRunner(), client)
    request = _enqueue(client)
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
