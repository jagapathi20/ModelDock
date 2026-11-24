"""Wire formats between API and worker (DESIGN.md section 6), and the Redis key names.

The API pushes a **request message** onto ``q:{model}:{version}``; the worker pushes a **reply
message** onto the request's ``reply_to`` list, ``replies:{api_instance_id}``. Both are JSON.

Both sides decode with pydantic, so a malformed message is rejected at the boundary with a
clear error instead of failing somewhere deeper. A process that receives one logs it and skips
it; it never crashes over one bad message (C1.19).
"""

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# --- Keys and timing constants ------------------------------------------------------------------

REPLY_PREFIX = "replies:"
# The worker sets this expiry on the reply list with every push (C1.18): whoever creates the key
# sets its expiry. Far above the 500 ms timeout; a reply older than this is useless anyway.
REPLY_TTL_S = 60
# Blocking pops wait at most this long, so loops wake at least once a second when idle (C1.19,
# C1.21): that's what lets a heartbeat show a loop is alive.
BLOCK_TIMEOUT_S = 1
# A loop whose last iteration is older than this is considered stuck or failed (C1.19).
HEARTBEAT_STALE_S = 3.0


def reply_key(api_instance_id: str) -> str:
    return f"{REPLY_PREFIX}{api_instance_id}"


def now_ms() -> float:
    """Wall-clock time in milliseconds since the Unix epoch, with sub-millisecond precision.

    A float, not a whole number: on one machine the whole path takes a few milliseconds, so
    whole-millisecond timestamps would round ``queue_wait_ms`` to 0 or 1 and make the overhead
    breakdown wrong by up to 1 ms (C2.5). A float64 holds today's epoch-milliseconds to about
    0.0002 ms.

    Wall clock, not ``time.monotonic()``: ``deadline_ms`` and ``enqueued_at_ms`` are written by
    the API and read by the worker, and a monotonic clock is only meaningful inside one process
    (its zero point is arbitrary). The cost is that comparing them assumes the two machines'
    clocks agree, which C2.3 records as a known limitation.
    """
    return time.time_ns() / 1_000_000


# --- Error codes (C2.1) -------------------------------------------------------------------------

INVALID_INPUT = "invalid_input"  # the runner raised InvalidInputError -> HTTP 422
INFERENCE_FAILED = "inference_failed"  # any other exception in inference -> HTTP 500
ErrorCode = Literal["invalid_input", "inference_failed"]


# --- Messages -------------------------------------------------------------------------------


class _Message(BaseModel):
    # Unknown fields are ignored, so a newer process can add a field without breaking an older
    # one during a rolling restart. Missing required fields are still an error.
    model_config = ConfigDict(extra="ignore", frozen=True)

    def encode(self) -> bytes:
        return self.model_dump_json().encode()

    @classmethod
    def decode(cls, raw: bytes | str):
        return cls.model_validate_json(raw)


class RequestMessage(_Message):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    # The worker pushes the reply to this key. Restricting it to replies:<32 hex> means a buggy
    # or forged message can't make the worker push into some other key, e.g. another queue.
    reply_to: str = Field(pattern=r"^replies:[0-9a-f]{32}$")
    model: str
    version: int
    input: dict[str, Any]
    deadline_ms: float  # wall clock; the worker drops the request once this has passed (C2.3)
    enqueued_at_ms: float  # wall clock; for queue_wait_ms now, and max_wait in Stage 3


class ReplyError(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: ErrorCode
    message: str


class Timings(BaseModel):
    """Where one request's time went inside the system (D13's latency stages, C3.5)."""

    model_config = ConfigDict(frozen=True)
    queue_wait_ms: float  # enqueued_at_ms -> popped by the worker (cross-process: wall clock)
    # popped -> the model call that produced this result started (worker's monotonic clock):
    # time spent in a forming batch, plus any re-runs after a failed batch.
    batch_wait_ms: float = 0.0
    inference_ms: float  # that model call, measured with perf_counter inside the worker
    batch_size: int  # requests in that model call


class ReplyMessage(_Message):
    request_id: str
    status: Literal["ok", "error"]
    output: dict[str, Any] | None = None
    error: ReplyError | None = None
    version: int
    timings: Timings | None = None
