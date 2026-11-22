"""The worker's main loop (Stage 2: one request at a time; Stage 3 adds batching).

Each iteration:

1. ``BLPOP`` the version's queue, waiting at most 1 s, so an idle loop still wakes every second
   (Stage 5 turns that into a heartbeat for ``/health/ready``, C1.21).
2. Decode the request. A malformed message is logged and dropped: it can't be answered (its
   ``reply_to`` can't be trusted) and must not stop the loop.
3. **Deadline check** (C2.3). If ``deadline_ms`` has passed, drop the request: no reply, count it.
   The API has already answered 504 (its timer and the deadline come from the same 500 ms), so a
   reply would be thrown away, and computing it would spend CPU on a client that is gone.
4. Run the model, timing it.
5. ``RPUSH`` the reply onto ``reply_to`` and ``EXPIRE`` it 60 s, in one pipeline (one round trip,
   C1.18): whoever creates the key sets its expiry, so a dead API process's list disappears.

Failure rules: **the loop never dies over one request.** Invalid input gets an ``invalid_input``
reply, any other inference exception an ``inference_failed`` reply, and a Redis error makes the
loop back off and retry. Only a bug in the loop itself can stop it.

Queue order: the API ``RPUSH``es (append at the tail) and the worker pops from the head, so the
queue is FIFO and Stage 3 can take the oldest *n* with one ``LPOP key n``.
"""

import logging
import time
from dataclasses import dataclass

import redis
from pydantic import ValidationError

from common.backoff import Backoff
from common.models import ModelSpec
from common.protocol import (
    BLOCK_TIMEOUT_S,
    INFERENCE_FAILED,
    INVALID_INPUT,
    REPLY_TTL_S,
    ReplyError,
    ReplyMessage,
    RequestMessage,
    Timings,
    now_ms,
)
from worker.runners import InvalidInputError, ModelRunner

log = logging.getLogger("modeldock.worker")


@dataclass
class WorkerStats:
    """Plain counters until Prometheus arrives in Stage 6."""

    processed: int = 0
    expired: int = 0  # dropped because the deadline had passed (D6)
    malformed: int = 0  # undecodable messages dropped
    invalid_input: int = 0
    inference_failed: int = 0
    redis_errors: int = 0


class Worker:
    def __init__(self, spec: ModelSpec, runner: ModelRunner, client: redis.Redis) -> None:
        self.spec = spec
        self.runner = runner
        self.redis = client
        self.stats = WorkerStats()
        self.last_loop_at = time.monotonic()  # Stage 5's /health/ready reads this
        self._stopping = False
        self._backoff = Backoff()

    # --- the loop ----------------------------------------------------------------------------

    def run(self) -> None:
        log.info("worker serving %s from %s", self.spec.queue, self.spec.artifact)
        while not self._stopping:
            try:
                self.run_once()
                self._backoff.reset()
            except redis.RedisError as e:
                # Redis down or restarting. Never exit: wait, then try again. Docker restarting
                # the worker wouldn't fix Redis, and the model would have to load again.
                self.stats.redis_errors += 1
                delay = self._backoff.next_delay()
                log.warning("redis error (%s); retrying in %.2f s", e, delay)
                time.sleep(delay)

    def stop(self) -> None:
        """Finish the current iteration, then return from run() (SIGTERM handling: Stage 5)."""
        self._stopping = True

    def run_once(self, block_timeout_s: float = BLOCK_TIMEOUT_S) -> bool:
        """One iteration. Returns True if a request was taken off the queue."""
        item = self.redis.blpop([self.spec.queue], timeout=block_timeout_s)
        self.last_loop_at = time.monotonic()
        if item is None:
            return False
        _key, raw = item
        try:
            request = RequestMessage.decode(raw)
        except ValidationError as e:
            self.stats.malformed += 1
            log.error("dropping malformed request message: %s", e.errors(include_url=False))
            return True
        self.handle(request)
        return True

    # --- one request --------------------------------------------------------------------------

    def handle(self, request: RequestMessage) -> None:
        popped_at = now_ms()
        # P9 asks for two checks, at pop and just before compute. Stage 2 computes right after
        # popping, so they're the same check. Stage 3's batching wait separates them.
        if self._expired(request, popped_at):
            return
        reply = self._infer(request, queue_wait_ms=popped_at - request.enqueued_at_ms)
        self._push(request.reply_to, reply)
        self.stats.processed += 1

    def _expired(self, request: RequestMessage, now: float) -> bool:
        if now < request.deadline_ms:
            return False
        self.stats.expired += 1
        log.info(
            "dropping expired request %s (%.1f ms past its deadline)",
            request.request_id,
            now - request.deadline_ms,
        )
        return True

    def _infer(self, request: RequestMessage, queue_wait_ms: float) -> ReplyMessage:
        t0 = time.perf_counter()
        output = error = None
        try:
            output = self.runner.predict_batch([request.input])[0]
        except InvalidInputError as e:
            self.stats.invalid_input += 1
            error = ReplyError(code=INVALID_INPUT, message=str(e))
        except Exception:
            # Anything else is our failure, not the client's: log the traceback here, and send
            # the client only a generic message (internals don't belong in an HTTP response).
            self.stats.inference_failed += 1
            log.exception("inference failed for request %s", request.request_id)
            error = ReplyError(code=INFERENCE_FAILED, message="inference failed")
        inference_ms = (time.perf_counter() - t0) * 1000
        return ReplyMessage(
            request_id=request.request_id,
            status="ok" if error is None else "error",
            output=output,
            error=error,
            version=self.spec.version,
            timings=Timings(queue_wait_ms=queue_wait_ms, inference_ms=inference_ms, batch_size=1),
        )

    def _push(self, reply_to: str, reply: ReplyMessage) -> None:
        # transaction=False: a plain pipeline sends both commands in one round trip without
        # MULTI/EXEC. Atomicity isn't needed: if the EXPIRE were lost, the next push sets it.
        pipe = self.redis.pipeline(transaction=False)
        pipe.rpush(reply_to, reply.encode())
        pipe.expire(reply_to, REPLY_TTL_S)
        pipe.execute()


def check_spec_matches(spec: ModelSpec, runner: ModelRunner) -> None:
    """Refuse to serve if the artifact disagrees with the catalog (C2.1).

    The API validates inputs against the catalog without loading the model (D3). If the catalog
    said 4 features and the artifact expected 5, the API would accept every request and the
    model would reject every one. Failing at startup turns that into a deploy-time error.
    """
    expected = spec.n_features
    actual = getattr(runner, "n_features", None)
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"{spec.name} v{spec.version}: catalog says {expected} features, "
            f"artifact {spec.artifact!r} expects {actual}"
        )
