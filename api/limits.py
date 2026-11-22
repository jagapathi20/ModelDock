"""Request body size limit (C2.2): bodies over the limit get 413 before they are parsed.

Why a middleware and not a field limit: by the time a handler (or FastAPI's validation) sees the
input, the whole body has already been read into memory and parsed as JSON. ``max_chars`` (422)
protects what comes after that (the Redis message, the tokenizer); this protects memory and
parsing.

It is raw ASGI middleware, not Starlette's ``BaseHTTPMiddleware``, because it must watch the body
*as it streams in*. An ASGI app receives the body as a sequence of ``http.request`` messages from
the ``receive`` callable; this middleware wraps ``receive`` and counts bytes.

Two cases:

- **Content-Length present** (almost every client): if it's over the limit, answer 413 at once
  without reading a byte of the body. The server (uvicorn) enforces Content-Length framing, so a
  client can't send more than it declared.
- **No Content-Length** (``Transfer-Encoding: chunked``): the size is unknown until the body ends,
  so count as chunks arrive. Once the count passes the limit, stop reading and answer 413. At
  most one chunk past the limit is ever held in memory.
"""

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _BodyTooLarge(Exception):
    pass


class BodySizeLimit:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared == "invalid":
            await _error(send, 400, "invalid_content_length", "Content-Length is not a number")
            return
        if declared is not None and declared > self.max_bytes:
            await self._too_large(send)
            return

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            # The app was reading the body, so it hasn't responded yet; if it somehow had, a
            # second response would be a protocol error, so re-raise and let the server close
            # the connection.
            if response_started:
                raise
            await self._too_large(send)

    async def _too_large(self, send: Send) -> None:
        await _error(send, 413, "body_too_large", f"request body exceeds {self.max_bytes} bytes")


def _content_length(scope: Scope) -> int | None | str:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return "invalid"
    return None


async def _error(send: Send, status: int, code: str, message: str) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                # The client may still be sending a body we won't read; closing the connection
                # stops it, instead of leaving unread bytes to be parsed as the next request.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
