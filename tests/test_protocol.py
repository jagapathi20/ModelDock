"""Wire formats (section 6, C2.1, C2.5)."""

import uuid

import pytest
from pydantic import ValidationError

from common.protocol import (
    ReplyError,
    ReplyMessage,
    RequestMessage,
    Timings,
    now_ms,
    reply_key,
)


def _request(**overrides) -> RequestMessage:
    fields = {
        "request_id": uuid.uuid4().hex,
        "reply_to": reply_key(uuid.uuid4().hex),
        "model": "iris",
        "version": 1,
        "input": {"features": [1, 2, 3, 4]},
        "deadline_ms": now_ms() + 500,
        "enqueued_at_ms": now_ms(),
    }
    return RequestMessage(**(fields | overrides))


def test_request_round_trip():
    msg = _request()
    assert RequestMessage.decode(msg.encode()) == msg


def test_reply_round_trip_with_error():
    reply = ReplyMessage(
        request_id=uuid.uuid4().hex,
        status="error",
        error=ReplyError(code="invalid_input", message="input 0: bad"),
        version=1,
        timings=Timings(queue_wait_ms=1.0, inference_ms=2.0, batch_size=1),
    )
    assert ReplyMessage.decode(reply.encode()) == reply


@pytest.mark.parametrize(
    "reply_to",
    ["q:sentiment:1", "replies:abc", "replies:" + "g" * 32, "other:" + "a" * 32],
)
def test_reply_to_must_be_a_reply_list(reply_to):
    # Otherwise a forged or buggy message could make the worker push into any key, e.g. a queue.
    with pytest.raises(ValidationError):
        _request(reply_to=reply_to)


def test_unknown_error_code_is_rejected():
    with pytest.raises(ValidationError):
        ReplyError(code="something_else", message="")


def test_unknown_fields_are_ignored():
    # A newer process may add fields; an older one must still read the message.
    raw = _request().model_dump() | {"priority": 5}
    assert RequestMessage.model_validate(raw).model == "iris"


def test_now_ms_is_wall_clock_milliseconds_with_fractions():
    import time

    assert abs(now_ms() - time.time() * 1000) < 50
    # Sub-millisecond precision: across many readings, not all are whole numbers.
    assert any(now_ms() % 1 for _ in range(100))
