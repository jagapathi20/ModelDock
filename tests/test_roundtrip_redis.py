"""End to end against a real Redis: API (in-process) -> Redis -> worker thread -> Redis -> API.

fakeredis reproduces the commands, but these tests check what only a real server shows: real
sockets and connection errors, real blocking pops across two client libraries (redis.asyncio
in the API, the sync client in the worker), and Redis's key-expiry rules that C1.18 relies on.

Marked ``redis``: they need a server at TEST_REDIS_URL (default database 15 of the Compose
Redis) and are skipped if none answers. Database 15 is flushed before and after each test.
"""

import asyncio
import threading

import httpx
import pytest
import redis

from api.main import create_app
from common.config import Settings
from common.models import IRIS_V1
from tests.conftest import TEST_REDIS_URL
from worker.loop import Worker


def _redis_available() -> bool:
    try:
        return redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=0.5).ping()
    except redis.RedisError:
        return False


needs_redis = pytest.mark.skipif(not _redis_available(), reason=f"no Redis at {TEST_REDIS_URL}")


@pytest.fixture
def real_redis():
    client = redis.Redis.from_url(TEST_REDIS_URL)
    client.flushdb()
    yield client
    client.flushdb()
    client.close()


@pytest.fixture
def iris_runner(tmp_path):
    pytest.importorskip("sklearn")
    from scripts.train_iris import train
    from worker.runners.tabular import SklearnClassifierRunner

    train(tmp_path)
    runner = SklearnClassifierRunner()
    runner.load(tmp_path)
    return runner


@pytest.fixture
def worker_thread(real_redis, iris_runner):
    """A real worker loop serving iris v1 in a background thread."""
    worker = Worker(IRIS_V1, iris_runner, redis.Redis.from_url(TEST_REDIS_URL, socket_timeout=2))
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    yield worker
    worker.stop()
    thread.join(timeout=3)  # the loop notices stop() within one BLPOP timeout (1 s)


async def _ready(client: httpx.AsyncClient, timeout_s: float = 3.0) -> bool:
    for _ in range(int(timeout_s / 0.05)):
        if (await client.get("/health/ready")).status_code == 200:
            return True
        await asyncio.sleep(0.05)
    return False


def _api(**overrides):
    return create_app(Settings(redis_url=TEST_REDIS_URL, **overrides))


@pytest.mark.redis
@needs_redis
async def test_full_round_trip_with_the_real_model(worker_thread, real_redis):
    app = _api()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert await _ready(client)
            setosa = await client.post(
                "/v1/models/iris/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}
            )
            virginica = await client.post(
                "/v1/models/iris/predict", json={"features": [6.7, 3.0, 5.2, 2.3]}
            )

    assert setosa.status_code == 200, setosa.text
    assert setosa.json()["output"]["label"] == "setosa"
    assert virginica.json()["output"]["label"] == "virginica"
    # The worker counts a request after pushing its reply, so the API can see the reply first.
    for _ in range(100):
        if worker_thread.stats.processed == 2:
            break
        await asyncio.sleep(0.01)
    assert worker_thread.stats.processed == 2


@pytest.mark.redis
@needs_redis
async def test_concurrent_requests_each_get_their_own_reply(worker_thread):
    # Many requests in flight from one API process share one reply list; the request_id is
    # what routes each reply to its own handler (D4).
    inputs = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3], [5.9, 3.0, 4.2, 1.5]] * 10
    expected = ["setosa", "virginica", "versicolor"] * 10
    app = _api(request_timeout_ms=2000)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert await _ready(client)
            responses = await asyncio.gather(
                *(client.post("/v1/models/iris/predict", json={"features": f}) for f in inputs)
            )
    assert [r.json()["output"]["label"] for r in responses] == expected


@pytest.mark.redis
@needs_redis
async def test_concurrent_requests_are_batched(real_redis, iris_runner):
    """With batching on, concurrent requests share model calls, and each still gets its answer."""
    from tests.test_worker import batching_spec

    worker = Worker(
        batching_spec(max_batch_size=4, max_wait_ms=20),
        iris_runner,
        redis.Redis.from_url(TEST_REDIS_URL, socket_timeout=2),
    )
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    inputs = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3], [5.9, 3.0, 4.2, 1.5]] * 8
    expected = ["setosa", "virginica", "versicolor"] * 8
    app = _api(request_timeout_ms=2000)
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert await _ready(client)
                responses = await asyncio.gather(
                    *(client.post("/v1/models/iris/predict", json={"features": f}) for f in inputs)
                )
    finally:
        worker.stop()
        thread.join(timeout=3)
    assert [r.json()["output"]["label"] for r in responses] == expected
    sizes = [int(r.headers["X-Batch-Size"]) for r in responses]
    assert max(sizes) > 1 and max(sizes) <= 4
    assert worker.stats.batches < len(inputs)


@pytest.mark.redis
@needs_redis
async def test_without_a_worker_the_request_times_out(real_redis):
    app = _api(request_timeout_ms=150)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/models/iris/predict", json={"features": [1, 2, 3, 4]})
    assert resp.status_code == 504
    # The request is still queued; when a worker starts it will see the deadline has passed
    # and drop it (C2.3), rather than compute an answer nobody is waiting for.
    assert real_redis.llen(IRIS_V1.queue) == 1


@pytest.mark.redis
@needs_redis
async def test_expired_request_left_in_the_queue_is_dropped(real_redis, iris_runner):
    app = _api(request_timeout_ms=100)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (
                await client.post("/v1/models/iris/predict", json={"features": [1, 2, 3, 4]})
            ).status_code == 504

    worker = Worker(IRIS_V1, iris_runner, redis.Redis.from_url(TEST_REDIS_URL))
    assert worker.run_once(block_timeout_s=0.5) == 1
    assert worker.stats.expired == 1 and worker.stats.processed == 0
    assert real_redis.keys("replies:*") == []


@pytest.mark.redis
@needs_redis
def test_reply_list_expiry_follows_redis_rules(real_redis, iris_runner):
    """The facts behind C1.18, checked on a real server."""
    key = "replies:" + "e" * 32
    # EXPIRE on a missing key does nothing and stores nothing for later.
    assert real_redis.expire(key, 60) is False
    # The worker's pipeline creates the key *with* its expiry.
    worker = Worker(IRIS_V1, iris_runner, real_redis)
    from common.protocol import ReplyMessage

    worker._push([(key, ReplyMessage(request_id="e" * 32, status="ok", output={}, version=1))])
    assert 0 < real_redis.ttl(key) <= 60
    # Popping the last element deletes the key (so a live listener's list rarely exists).
    real_redis.lpop(key)
    assert real_redis.exists(key) == 0


async def test_api_starts_without_redis_and_reports_not_ready():
    # Port 1 refuses connections: Redis "down". The API must still start, stay live, report
    # not ready, and answer predict with 503 (enqueue failed), not hang or crash.
    app = create_app(Settings(redis_url="redis://127.0.0.1:1/0", request_timeout_ms=300))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await asyncio.sleep(0.2)
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 503
            resp = await client.post("/v1/models/iris/predict", json={"features": [1, 2, 3, 4]})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "backend_unavailable"
