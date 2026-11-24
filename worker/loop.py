"""The worker's main loop: dynamic batching, pull-when-ready (D5, C1.11, C3.1).

One iteration forms and runs one batch:

1. **Wait for the first request:** ``BLPOP`` the version's queue, at most 1 s, so an idle loop
   still wakes every second (the heartbeat Stage 5's ``/health/ready`` reads, C1.21).
2. **Take what's already waiting:** ``LPOP key count`` for up to ``max_batch_size - 1`` more, in
   one round trip (Redis >= 6.2).
3. **If the batch isn't full, wait for more**, but only until ``max_wait`` after the first
   request arrived (C3.1: measured from its ``enqueued_at_ms``, clamped to [0, max_wait] so clock
   skew can't stretch it). The worker keeps that time itself: it sleeps in short steps and polls
   with a non-blocking ``LPOP``. It can't use a ``BLPOP`` timeout for this, because Redis checks
   blocking timeouts only on its internal timer (``hz``, 10 per second by default), so a 15 ms
   timeout can take about 100 ms (C3.1).
4. **Check deadlines again just before compute** and drop the expired (C2.3).
5. **Run the batch**, with the failure rules of C3.3: one bad input fails only its own request.
6. **Reply to every request** in one pipeline: ``RPUSH`` + ``EXPIRE 60`` per reply (C1.18).

There is no collector thread: the next batch is formed only when this one is done (C1.11), so
every waiting request stays in Redis, where the API's queue-length check can see it.

Each request is checked as it is popped: decoded (malformed messages are dropped), deadline
checked (C2.3), and validated against the catalog schema (C3.3). A request that fails
validation gets its ``invalid_input`` reply without ever reaching the model.

**The loop never dies over one request.** Invalid input gets ``invalid_input``, any other
inference exception ``inference_failed``, and a Redis error makes the loop back off and retry.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

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

# How often an open batch checks the queue for newcomers (C3.1). A batch closes at most this long
# after it fills. Each step costs one LPOP round trip, so at light load a 15 ms max_wait costs
# up to 15 cheap Redis calls; under load batches fill on the first LPOP and never poll.
POLL_INTERVAL_S = 0.001


@dataclass
class WorkerStats:
    """Plain counters until Prometheus arrives in Stage 6."""

    processed: int = 0  # requests replied to (ok or error)
    batches: int = 0  # model calls on a formed batch
    expired: int = 0  # dropped because the deadline had passed (C2.3)
    malformed: int = 0  # undecodable messages dropped
    invalid_input: int = 0
    inference_failed: int = 0
    rerun_alone: int = 0  # batches that failed and were re-run one request at a time (C3.3)
    redis_errors: int = 0
    batch_sizes: dict[int, int] = field(default_factory=dict)  # size -> batches


@dataclass
class Pending:
    """A request taken off the queue, with when the worker took it."""

    request: RequestMessage
    popped_mono: float  # worker's monotonic clock: for batch_wait_ms
    popped_wall: float  # wall clock, comparable with the API's enqueued_at_ms


@dataclass
class Outcome:
    """The result for one request, and the model call that produced it."""

    output: dict[str, Any] | None = None
    error: ReplyError | None = None
    started_mono: float = 0.0  # when the model call that produced this result started
    inference_ms: float = 0.0
    batch_size: int = 0


def batch_window_s(max_wait_ms: float, enqueued_at_ms: float, now_wall_ms: float) -> float:
    """How long, from now, the batch may stay open for more requests (C3.1).

    ``max_wait`` runs from the first request's arrival (D5: no request waits more than
    ``max_wait`` for its batch to start), so time it already spent queued counts against it. The
    queue time comes from two clocks, the API's and the worker's, so it's clamped to
    ``[0, max_wait]``: a worker clock running ahead can at most close the batch at once, and one
    running behind can at most give the full ``max_wait``, never more.
    """
    queued_ms = min(max(now_wall_ms - enqueued_at_ms, 0.0), max_wait_ms)
    return (max_wait_ms - queued_ms) / 1000


class Worker:
    def __init__(self, spec: ModelSpec, runner: ModelRunner, client: redis.Redis) -> None:
        self.spec = spec
        self.runner = runner
        self.redis = client
        self.stats = WorkerStats()
        self.last_loop_at = time.monotonic()  # Stage 5's /health/ready reads this
        self._stopping = False
        self._backoff = Backoff()

    # --- the loop ------------------------------------------------------------------------------

    def run(self) -> None:
        log.info(
            "worker serving %s from %s (max_batch_size=%d, max_wait=%g ms)",
            self.spec.queue,
            self.spec.artifact,
            self.spec.max_batch_size,
            self.spec.max_wait_ms,
        )
        while not self._stopping:
            try:
                self.run_once()
                self._backoff.reset()
            except redis.RedisError as e:
                # Redis down or restarting. Never exit: wait, then try again. Restarting the
                # worker wouldn't fix Redis, and the model would have to load again.
                self.stats.redis_errors += 1
                delay = self._backoff.next_delay()
                log.warning("redis error (%s); retrying in %.2f s", e, delay)
                time.sleep(delay)

    def stop(self) -> None:
        """Finish the current iteration, then return from run() (SIGTERM handling: Stage 5)."""
        self._stopping = True

    def run_once(self, block_timeout_s: float = BLOCK_TIMEOUT_S) -> int:
        """Form and run one batch. Returns how many messages were taken off the queue."""
        item = self.redis.blpop([self.spec.queue], timeout=block_timeout_s)
        self.last_loop_at = time.monotonic()
        if item is None:
            return 0

        batch: list[Pending] = []
        replies: list[tuple[str, ReplyMessage]] = []  # invalid inputs answered without the model
        taken = self._take([item[1]], batch, replies)
        taken += self._collect(batch, replies)

        outcomes = self._compute(batch)
        for p in batch:
            outcome = outcomes.get(p.request.request_id)
            if outcome is not None:  # None: expired at the pre-compute check, no reply
                replies.append((p.request.reply_to, self._reply(p, outcome)))
        self._push(replies)
        self.stats.processed += len(replies)
        return taken

    # --- forming the batch -----------------------------------------------------------------------

    def _collect(self, batch: list[Pending], replies: list[tuple[str, ReplyMessage]]) -> int:
        """Fill the batch: first what's queued already, then newcomers until the window closes.

        The window starts from the batch's first *valid* request: if the very first message was
        malformed, expired or invalid, the next one that isn't anchors it.
        """
        size = self.spec.max_batch_size
        taken = 0
        if size > 1:
            taken += self._take(self._lpop(size - len(batch)), batch, replies)
        close_at = self._close_at(batch)
        while len(batch) < size:
            if close_at is None:  # nothing valid yet: anchor on the next request, if any
                more = self._lpop(size - len(batch))
                if not more:
                    break
                taken += self._take(more, batch, replies)
                close_at = self._close_at(batch)
                continue
            left = close_at - time.monotonic()
            if left <= 0:
                break
            time.sleep(min(POLL_INTERVAL_S, left))
            taken += self._take(self._lpop(size - len(batch)), batch, replies)
        return taken

    def _close_at(self, batch: list[Pending]) -> float | None:
        if not batch:
            return None
        first = batch[0]
        return first.popped_mono + batch_window_s(
            self.spec.max_wait_ms, first.request.enqueued_at_ms, first.popped_wall
        )

    def _lpop(self, count: int) -> list[bytes]:
        if count <= 0:
            return []
        return self.redis.lpop(self.spec.queue, count) or []  # None when the list is empty

    def _take(
        self, raws: list[bytes], batch: list[Pending], replies: list[tuple[str, ReplyMessage]]
    ) -> int:
        """Decode, deadline-check and validate popped messages; add the good ones to the batch."""
        for raw in raws:
            popped_mono, popped_wall = time.monotonic(), now_ms()
            try:
                request = RequestMessage.decode(raw)
            except ValidationError as e:
                # Unanswerable: its reply_to is exactly the part that can't be trusted.
                self.stats.malformed += 1
                log.error("dropping malformed request message: %s", e.errors(include_url=False))
                continue
            if self._expired(request, popped_wall):  # first deadline check: at pop (C2.3)
                continue
            pending = Pending(request, popped_mono, popped_wall)
            try:
                # Same schema as the API (C3.3): catches anything that reached the queue some
                # other way, so one bad input can't share a batch with good ones.
                request = request.model_copy(
                    update={"input": self.spec.validate_input(request.input)}
                )
                pending.request = request
            except ValidationError as e:
                self.stats.invalid_input += 1
                outcome = Outcome(
                    error=ReplyError(code=INVALID_INPUT, message=_summarize(e)),
                    started_mono=time.monotonic(),
                )
                replies.append((request.reply_to, self._reply(pending, outcome)))
                continue
            batch.append(pending)
        return len(raws)

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

    # --- running the batch (C3.3) ---------------------------------------------------------------

    def _compute(self, batch: list[Pending]) -> dict[str, Outcome]:
        """Run the batch; return an outcome per request id (expired requests get none).

        - Second deadline check before every model call (C2.3): batching may have kept a request
          waiting past its deadline.
        - ``InvalidInputError`` naming an input: that request gets ``invalid_input``; the rest
          run again as one batch.
        - Any other failure (or ``InvalidInputError`` without a usable index): run each request
          alone, so each gets its own result. One that fails alone too gets the error.
        """
        outcomes: dict[str, Outcome] = {}
        todo = batch
        while todo:
            todo = self._drop_expired(todo)
            if not todo:
                break
            started, t0 = time.monotonic(), time.perf_counter()
            try:
                outputs = self.runner.predict_batch([p.request.input for p in todo])
                if len(outputs) != len(todo):
                    raise RuntimeError(f"runner returned {len(outputs)} outputs for {len(todo)}")
            except InvalidInputError as e:
                ms = (time.perf_counter() - t0) * 1000
                if e.index is not None and 0 <= e.index < len(todo):
                    bad = todo[e.index]
                    self.stats.invalid_input += 1
                    outcomes[bad.request.request_id] = Outcome(
                        error=ReplyError(code=INVALID_INPUT, message=str(e)),
                        started_mono=started,
                        inference_ms=ms,
                        batch_size=len(todo),
                    )
                    todo = todo[: e.index] + todo[e.index + 1 :]
                    continue
                outcomes.update(self._run_alone(todo))
                break
            except Exception:
                log.exception("batch of %d failed; re-running each request alone", len(todo))
                outcomes.update(self._run_alone(todo))
                break
            ms = (time.perf_counter() - t0) * 1000
            self._count_batch(len(todo))
            for p, output in zip(todo, outputs, strict=True):
                outcomes[p.request.request_id] = Outcome(
                    output=output, started_mono=started, inference_ms=ms, batch_size=len(todo)
                )
            break
        return outcomes

    def _run_alone(self, batch: list[Pending]) -> dict[str, Outcome]:
        if len(batch) > 1:
            self.stats.rerun_alone += 1
        outcomes = {}
        for p in batch:
            if self._expired(p.request, now_ms()):
                continue
            started, t0 = time.monotonic(), time.perf_counter()
            output = error = None
            try:
                output = self.runner.predict_batch([p.request.input])[0]
                self._count_batch(1)
            except InvalidInputError as e:
                self.stats.invalid_input += 1
                error = ReplyError(code=INVALID_INPUT, message=str(e))
            except Exception:
                # Our failure, not the client's: traceback here, a generic message to the client.
                self.stats.inference_failed += 1
                log.exception("inference failed for request %s", p.request.request_id)
                error = ReplyError(code=INFERENCE_FAILED, message="inference failed")
            outcomes[p.request.request_id] = Outcome(
                output=output,
                error=error,
                started_mono=started,
                inference_ms=(time.perf_counter() - t0) * 1000,
                batch_size=1,
            )
        return outcomes

    def _drop_expired(self, todo: list[Pending]) -> list[Pending]:
        now = now_ms()
        return [p for p in todo if not self._expired(p.request, now)]

    def _count_batch(self, size: int) -> None:
        self.stats.batches += 1
        self.stats.batch_sizes[size] = self.stats.batch_sizes.get(size, 0) + 1

    # --- replying -------------------------------------------------------------------------------

    def _reply(self, p: Pending, outcome: Outcome) -> ReplyMessage:
        return ReplyMessage(
            request_id=p.request.request_id,
            status="ok" if outcome.error is None else "error",
            output=outcome.output,
            error=outcome.error,
            version=self.spec.version,
            timings=Timings(
                queue_wait_ms=p.popped_wall - p.request.enqueued_at_ms,
                batch_wait_ms=max(outcome.started_mono - p.popped_mono, 0.0) * 1000,
                inference_ms=outcome.inference_ms,
                batch_size=outcome.batch_size,
            ),
        )

    def _push(self, replies: list[tuple[str, ReplyMessage]]) -> None:
        """All replies of a batch in one round trip: RPUSH + EXPIRE per reply (C1.18).

        transaction=False: a plain pipeline, no MULTI/EXEC. Atomicity isn't needed: if an EXPIRE
        were lost, the next push to that list sets it.
        """
        if not replies:
            return
        pipe = self.redis.pipeline(transaction=False)
        for reply_to, reply in replies:
            pipe.rpush(reply_to, reply.encode())
            pipe.expire(reply_to, REPLY_TTL_S)
        pipe.execute()


def _summarize(e: ValidationError) -> str:
    first = e.errors(include_url=False)[0]
    where = ".".join(str(x) for x in first["loc"]) or "input"
    return f"{where}: {first['msg']}"


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
