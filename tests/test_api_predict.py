"""The predict endpoint and the RPC machinery behind it (D4, C1.16, C1.18, C1.19, C2.1–C2.5).

A stand-in worker (``answer``) pops the request from the fake Redis and pushes whatever reply a
test wants, so each test controls exactly what comes back, and when.
"""

import asyncio

import fakeredis
import pytest

from common.models import IRIS_V1, SENTIMENT_V1
from common.protocol import ReplyError, ReplyMessage, RequestMessage, Timings

IRIS_INPUT = {"features": [5.1, 3.5, 1.4, 0.2]}


def _ok(request: RequestMessage) -> ReplyMessage:
    return ReplyMessage(
        request_id=request.request_id,
        status="ok",
        output={"label": "setosa", "score": 0.97},
        version=request.version,
        timings=Timings(queue_wait_ms=0.5, inference_ms=1.25, batch_size=1),
    )


def _err(code: str):
    def reply(request: RequestMessage) -> ReplyMessage:
        return ReplyMessage(
            request_id=request.request_id,
            status="error",
            error=ReplyError(code=code, message="details"),
            version=request.version,
        )

    return reply


async def answer(fake_server, queue: str, make_reply=_ok, delay_s: float = 0.0) -> RequestMessage:
    """Act as the worker for one request; returns the request it saw."""
    r = fakeredis.FakeAsyncRedis(server=fake_server)
    _, raw = await r.blpop([queue], timeout=2)
    request = RequestMessage.decode(raw)
    await asyncio.sleep(delay_s)
    await r.rpush(request.reply_to, make_reply(request).encode())
    return request


async def test_round_trip_ok(make_api, fake_server):
    async with make_api() as (client, app):
        worker = asyncio.create_task(answer(fake_server, IRIS_V1.queue))
        resp = await client.post("/v1/models/iris/predict", json=IRIS_INPUT)
        request = await worker

    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "request_id": request.request_id,
        "model": "iris",
        "version": 1,
        "output": {"label": "setosa", "score": 0.97},
    }
    assert resp.headers["X-Request-Id"] == request.request_id
    assert resp.headers["X-Model-Version"] == "1"
    assert "queue;dur=0.500" in resp.headers["Server-Timing"]
    assert "inference;dur=1.250" in resp.headers["Server-Timing"]
    assert "batch;dur=" in resp.headers["Server-Timing"]
    assert resp.headers["X-Batch-Size"] == "1"
    assert len(app.state.pending) == 0  # cleaned up in finally


async def test_request_message_carries_deadline_and_reply_to(make_api, fake_server):
    async with make_api(request_timeout_ms=500) as (client, app):
        worker = asyncio.create_task(answer(fake_server, IRIS_V1.queue))
        await client.post("/v1/models/iris/predict", json=IRIS_INPUT)
        request = await worker

    # One clock reading sets both (C1.16).
    assert request.deadline_ms - request.enqueued_at_ms == pytest.approx(500)
    assert request.reply_to == app.state.listener.key
    assert request.input == {"features": [5.1, 3.5, 1.4, 0.2]}


async def test_no_worker_means_504_and_no_leak(make_api):
    async with make_api(request_timeout_ms=100) as (client, app):
        resp = await client.post("/v1/models/iris/predict", json=IRIS_INPUT)
        assert len(app.state.pending) == 0

    assert resp.status_code == 504
    assert resp.json()["error"]["code"] == "timeout"
    assert len(resp.json()["error"]["request_id"]) == 32


async def test_late_reply_is_discarded(make_api, fake_server):
    async with make_api(request_timeout_ms=100) as (client, app):
        worker = asyncio.create_task(answer(fake_server, IRIS_V1.queue, delay_s=0.25))
        resp = await client.post("/v1/models/iris/predict", json=IRIS_INPUT)
        assert resp.status_code == 504
        await worker  # the reply lands after the handler gave up
        for _ in range(50):  # give the listener a moment to read it
            if app.state.listener.stats.late:
                break
            await asyncio.sleep(0.02)

        assert app.state.listener.stats.late == 1
        assert app.state.listener.healthy()  # the late reply didn't hurt the listener
        # ...and the next request is still served normally.
        worker = asyncio.create_task(answer(fake_server, IRIS_V1.queue))
        assert (await client.post("/v1/models/iris/predict", json=IRIS_INPUT)).status_code == 200
        await worker


@pytest.mark.parametrize(("code", "status"), [("invalid_input", 422), ("inference_failed", 500)])
async def test_worker_errors_map_to_http_status(make_api, fake_server, code, status):
    async with make_api() as (client, _):
        worker = asyncio.create_task(answer(fake_server, IRIS_V1.queue, make_reply=_err(code)))
        resp = await client.post("/v1/models/iris/predict", json=IRIS_INPUT)
        request = await worker

    assert resp.status_code == status
    assert resp.json()["error"] == {
        "code": code,
        "message": "details",
        "request_id": request.request_id,
    }


@pytest.mark.parametrize(
    ("model", "body"),
    [
        ("iris", {"features": [1, 2, 3]}),
        ("iris", {"features": ["a", 2, 3, 4]}),
        ("sentiment", {"text": "x" * (SENTIMENT_V1.max_chars + 1)}),
        ("sentiment", {"txt": "hello"}),
    ],
)
async def test_invalid_input_is_rejected_before_enqueue(make_api, fake_server, model, body):
    async with make_api() as (client, _):
        resp = await client.post(f"/v1/models/{model}/predict", json=body)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_input"
    assert resp.json()["error"]["details"]
    # Never queued: nothing for a worker to choke on.
    assert fakeredis.FakeRedis(server=fake_server).llen(f"q:{model}:1") == 0


async def test_invalid_json_is_422(make_api):
    async with make_api() as (client, _):
        resp = await client.post(
            "/v1/models/iris/predict",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 422


async def test_unknown_model_is_404(make_api):
    async with make_api() as (client, _):
        resp = await client.post("/v1/models/nope/predict", json={"text": "hi"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "unknown_model"


async def test_body_over_limit_is_413(make_api):
    async with make_api(max_body_bytes=64) as (client, _):
        resp = await client.post("/v1/models/sentiment/predict", json={"text": "x" * 100})
    assert resp.status_code == 413


async def test_enqueue_failure_is_503_and_cleans_up(make_api, monkeypatch):
    async with make_api() as (client, app):

        async def broken_rpush(*args, **kwargs):
            raise ConnectionError("Redis is down")

        monkeypatch.setattr(app.state.rpc._redis, "rpush", broken_rpush)
        resp = await client.post("/v1/models/iris/predict", json=IRIS_INPUT)
        assert len(app.state.pending) == 0  # no leak on the failure path (C1.19)

    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "1"
    assert resp.json()["error"]["code"] == "backend_unavailable"


async def test_ready_after_listener_runs(make_api):
    async with make_api() as (client, app):
        for _ in range(50):
            if app.state.listener.healthy():
                break
            await asyncio.sleep(0.02)
        resp = await client.get("/health/ready")
    assert resp.status_code == 200


async def test_not_ready_without_lifespan():
    # An app whose listener never started (here: lifespan not run) must not claim readiness.
    import httpx

    from api.main import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503


async def test_largest_valid_text_fits_the_body_limit(make_api):
    """1,000 characters that JSON escapes as long as possible must still pass the 16 KB limit.

    An emoji outside the Basic Multilingual Plane is one character to the schema but 12 bytes
    in JSON (a surrogate pair, \\ud83d\\ude00), the worst case. So the largest valid sentiment
    body is about 12 KB, and 16 KB never rejects a request the schema would accept (C2.2).
    """
    import json

    body = json.dumps({"text": "\U0001f600" * SENTIMENT_V1.max_chars}).encode()
    assert len(body) < 16_384
    async with make_api(request_timeout_ms=50) as (client, _):
        resp = await client.post(
            "/v1/models/sentiment/predict",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 504  # passed both limits and was queued; no worker in this test
