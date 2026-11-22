"""The reply listener supervises itself (C1.19)."""

import asyncio
import uuid

import fakeredis
import pytest
import redis

from api.rpc import PendingReplies, ReplyListener
from common.protocol import HEARTBEAT_STALE_S, ReplyMessage


def _reply(request_id: str) -> ReplyMessage:
    return ReplyMessage(request_id=request_id, status="ok", output={"label": "x"}, version=1)


@pytest.fixture
def key():
    return f"replies:{uuid.uuid4().hex}"


async def _until(cond, timeout_s: float = 2.0) -> None:
    for _ in range(int(timeout_s / 0.01)):
        if cond():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not reached")


async def test_resolves_the_matching_future(fake_server, key):
    pending = PendingReplies()
    listener = ReplyListener(fakeredis.FakeAsyncRedis(server=fake_server), key, pending)
    listener.start()
    try:
        fut = pending.register("a" * 32)
        await fakeredis.FakeAsyncRedis(server=fake_server).rpush(key, _reply("a" * 32).encode())
        reply = await asyncio.wait_for(fut, 2)
        assert reply.request_id == "a" * 32
    finally:
        await listener.stop()


async def test_malformed_reply_is_skipped_and_listener_continues(fake_server, key):
    pending = PendingReplies()
    listener = ReplyListener(fakeredis.FakeAsyncRedis(server=fake_server), key, pending)
    listener.start()
    try:
        producer = fakeredis.FakeAsyncRedis(server=fake_server)
        fut = pending.register("b" * 32)
        await producer.rpush(key, b"{garbage", _reply("b" * 32).encode())
        assert (await asyncio.wait_for(fut, 2)).request_id == "b" * 32
        assert listener.stats.malformed == 1
    finally:
        await listener.stop()


async def test_reply_for_a_finished_future_does_not_raise():
    # The handler timed out (future cancelled) just as its reply arrived. set_result would raise
    # InvalidStateError; resolve() must report "late" instead (C1.19).
    pending = PendingReplies()
    fut = pending.register("c" * 32)
    fut.cancel()
    assert pending.resolve(_reply("c" * 32)) is False


class _FlakyRedis:
    """BLPOP fails `failures` times with a connection error, then behaves like a fake Redis."""

    def __init__(self, server, failures: int) -> None:
        self._real = fakeredis.FakeAsyncRedis(server=server)
        self.failures = failures

    async def blpop(self, keys, timeout):  # noqa: ASYNC109 - mirrors redis-py's signature
        if self.failures > 0:
            self.failures -= 1
            raise redis.ConnectionError("Connection refused")
        return await self._real.blpop(keys, timeout=timeout)


async def test_survives_connection_errors_and_recovers(fake_server, key, monkeypatch):
    monkeypatch.setattr("api.rpc.Backoff.next_delay", lambda self: 0.01)
    pending = PendingReplies()
    listener = ReplyListener(_FlakyRedis(fake_server, failures=3), key, pending)
    listener.start()
    try:
        fut = pending.register("d" * 32)
        await fakeredis.FakeAsyncRedis(server=fake_server).rpush(key, _reply("d" * 32).encode())
        assert (await asyncio.wait_for(fut, 2)).request_id == "d" * 32
        assert listener.stats.redis_errors == 3
        assert listener._task is not None and not listener._task.done()  # still running
    finally:
        await listener.stop()


async def test_heartbeat_drives_health(fake_server, key):
    listener = ReplyListener(fakeredis.FakeAsyncRedis(server=fake_server), key, PendingReplies())
    assert listener.healthy() is False  # never ran
    listener.start()
    try:
        await _until(lambda: listener.last_iteration is not None)
        assert listener.healthy()
        # A heartbeat older than 3 s means stuck or failing: not ready.
        assert listener.healthy(now=listener.last_iteration + HEARTBEAT_STALE_S + 0.1) is False
    finally:
        await listener.stop()


async def test_stop_cancels_the_task(fake_server, key):
    listener = ReplyListener(fakeredis.FakeAsyncRedis(server=fake_server), key, PendingReplies())
    listener.start()
    await listener.stop()
    assert listener._task.cancelled() or listener._task.done()
