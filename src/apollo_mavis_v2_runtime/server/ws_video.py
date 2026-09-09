"""/ws/video/{stream_id} + /video/{stream_id}.mjpg (04-runtime §13.4).

WS frames are the shared encoded buffer (12-byte ``<dI`` header + JPEG);
MJPEG strips the header and wraps multipart — encode-once, byte-identical
payloads. Unknown stream ids (and reserved "sim"/"twin" outside a session)
close 1008.
"""

from __future__ import annotations

import asyncio

from apollo_mavis_v2_core.protocol import HEADER_SIZE
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from starlette.requests import Request
from starlette.responses import Response

_POLL_S = 0.005  # cheap slot poll; << frame period at <= 30 fps
_BOUNDARY = b"apollo-frame"


async def endpoint(websocket: WebSocket) -> None:
    runtime = websocket.app.state.runtime
    stream_id = websocket.path_params["stream_id"]
    await websocket.accept()
    hub = runtime.hub
    if not hub.has(stream_id):
        await websocket.close(code=1008)
        return
    slot = hub.slot(stream_id)
    last_put = 0.0
    try:
        while True:
            if not hub.has(stream_id):  # stream removed (session teardown)
                await websocket.close(code=1008)
                return
            got = slot.get()
            if got is None or got[1] == last_put:
                await asyncio.sleep(_POLL_S)
                continue
            buf, last_put = got  # depth-1 latest: stalls skip to newest frame
            await websocket.send_bytes(buf)
    except WebSocketDisconnect:
        pass


async def mjpeg(request: Request) -> Response:
    runtime = request.app.state.runtime
    stream_id = request.path_params["stream_id"]
    hub = runtime.hub
    if not hub.has(stream_id):
        return Response(status_code=404, content=f"unknown stream {stream_id!r}")
    slot = hub.slot(stream_id)

    async def gen():
        last_put = 0.0
        while hub.has(stream_id):
            got = slot.get()
            if got is None or got[1] == last_put:
                await asyncio.sleep(_POLL_S)
                continue
            buf, last_put = got
            jpeg = buf[HEADER_SIZE:]  # same encoded payload as the WS path
            yield (
                b"--" + _BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            )

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
    )


__all__ = ["endpoint", "mjpeg"]
