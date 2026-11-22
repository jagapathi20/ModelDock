"""RPC over Redis lists: the API side (D4, C1.18, C1.19).

A request travels like this:

    handler ──RPUSH q:{model}:{version}──► worker
       ▲                                     │
       │ future resolved                     │ RPUSH replies:{api_instance_id} + EXPIRE
       │                                     ▼
    ReplyListener ◄──BLPOP replies:{api_instance_id}

- **PendingReplies** maps ``request_id → asyncio.Future``. The handler creates the future, the
  listener resolves it.
- **ReplyListener** is one background task per API process, looping on ``BLPOP`` of this
  process's reply list. It supervises itself: it catches every error, backs off, and carries on.
  Its heartbeat feeds ``/health/ready``.
- **RpcClient.call()** is what a handler awaits: register, enqueue, wait with a timeout, and
  always unregister in ``finally``.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

from pydantic import ValidationError
from redis.asyncio import Redis

from common.backoff import Backoff
from common.models import ModelSpec
from common.protocol import (
    BLOCK_TIMEOUT_S,
    HEARTBEAT_STALE_S,
    ReplyMessage,
    RequestMessage,
    now_ms,
    reply_key,
)

log = logging.getLogger("modeldock.api.rpc")


class RequestTimeout(Exception):
    """No reply within the request timeout (-> 504)."""


class EnqueueFailed(Exception):
    """Redis refused or failed the enqueue (-> 503)."""


class PendingReplies:
    """``request_id → Future`` for the requests this process is waiting on.

    Only the event loop thread touches it (the handler and the listener are both coroutines on
    the same loop), so it needs no lock: between two ``await``s, nothing else runs.
    """

    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future[ReplyMessage]] = {}

    def register(self, request_id: str) -> asyncio.Future[ReplyMessage]:
        fut = asyncio.get_running_loop().create_future()
        self._futures[request_id] = fut
        return fut

    def discard(self, request_id: str) -> None:
        # pop(..., None): doesn't raise if the entry is already gone (C1.19).
        self._futures.pop(request_id, None)

    def resolve(self, reply: ReplyMessage) -> bool:
        """Hand a reply to its waiting handler. False if nobody is waiting (a late reply)."""
        fut = self._futures.get(reply.request_id)
        # done(): the handler may have timed out (or been cancelled) at the very moment this
        # reply arrived. set_result on a finished future raises InvalidStateError, which would
        # kill the listener if it escaped (C1.19).
        if fut is None or fut.done():
            return False
        fut.set_result(reply)
        return True

    def __len__(self) -> int:
        return len(self._futures)


@dataclass
class ListenerStats:
    replies: int = 0
    late: int = 0  # reply arrived after its handler gave up (504), discarded (D4)
    malformed: int = 0
    redis_errors: int = 0


class ReplyListener:
    """The one task that reads this process's reply list.

    It never exits on an error: a connection error means back off and retry; a bad message
    means log it and skip it. If it did exit, every later request in this process would wait
    for a reply nobody reads and time out with 504, while ``/health/live`` still said 200: a
    silent failure (C1.19). The heartbeat (``last_iteration``) catches the remaining cases, a
    loop that is stuck or keeps failing, through ``/health/ready``.
    """

    def __init__(self, client: Redis, key: str, pending: PendingReplies) -> None:
        # A dedicated client (its own connection pool): BLPOP holds its connection for up to a
        # second at a time, which would otherwise starve handlers of pooled connections.
        self._redis = client
        self.key = key
        self._pending = pending
        self._backoff = Backoff()
        self.stats = ListenerStats()
        self.last_iteration: float | None = None  # monotonic; None until the first BLPOP returns
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"reply-listener:{self.key}")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def healthy(self, now: float | None = None) -> bool:
        if self.last_iteration is None:
            return False
        now = time.monotonic() if now is None else now
        return now - self.last_iteration < HEARTBEAT_STALE_S

    async def _run(self) -> None:
        while True:
            try:
                item = await self._redis.blpop([self.key], timeout=BLOCK_TIMEOUT_S)
            except Exception as e:  # CancelledError is not an Exception, so stop() still works
                # Typically ConnectionError or TimeoutError while Redis restarts. No heartbeat
                # update here, so /health/ready turns false after 3 s of failures.
                self.stats.redis_errors += 1
                delay = self._backoff.next_delay()
                log.warning(
                    "reply listener: %s: %s; retrying in %.2f s", type(e).__name__, e, delay
                )
                await asyncio.sleep(delay)
                continue
            self._backoff.reset()
            self.last_iteration = time.monotonic()
            if item is not None:
                self._dispatch(item[1])

    def _dispatch(self, raw: bytes) -> None:
        try:
            reply = ReplyMessage.decode(raw)
        except ValidationError as e:
            self.stats.malformed += 1
            log.error("reply listener: dropping malformed reply: %s", e.errors(include_url=False))
            return
        self.stats.replies += 1
        if not self._pending.resolve(reply):
            self.stats.late += 1
            log.info("reply listener: discarding late reply for %s", reply.request_id)


class RpcClient:
    """Sends one request to a model version's queue and waits for its reply."""

    def __init__(
        self, client: Redis, api_instance_id: str, pending: PendingReplies, timeout_ms: int
    ) -> None:
        self._redis = client
        self.reply_to = reply_key(api_instance_id)
        self._pending = pending
        self.timeout_ms = timeout_ms

    async def call(self, spec: ModelSpec, model_input: dict) -> ReplyMessage:
        """Raises RequestTimeout (504) or EnqueueFailed (503)."""
        # One clock reading sets both the deadline the worker checks and this handler's timer,
        # so they expire together (C1.16). (Cross-process, the worker compares deadline_ms to its
        # own wall clock: C2.3.)
        arrived = now_ms()
        request = RequestMessage(
            request_id=uuid.uuid4().hex,
            reply_to=self.reply_to,
            model=spec.name,
            version=spec.version,
            input=model_input,
            deadline_ms=arrived + self.timeout_ms,
            enqueued_at_ms=arrived,
        )
        # Register *before* enqueueing: a fast worker could otherwise reply before the future
        # exists, and the reply would be discarded as late.
        fut = self._pending.register(request.request_id)
        try:
            async with asyncio.timeout(self.timeout_ms / 1000):
                try:
                    await self._redis.rpush(spec.queue, request.encode())
                except Exception as e:
                    raise EnqueueFailed(f"{type(e).__name__}: {e}") from e
                return await fut
        except TimeoutError:
            raise RequestTimeout(request.request_id) from None
        finally:
            # Every exit path cleans up: success, timeout, failed enqueue, and cancellation (the
            # client disconnected). Cleaning up only on timeout would leak one entry per request
            # during a Redis outage, exactly when requests fail fastest (C1.19).
            self._pending.discard(request.request_id)
