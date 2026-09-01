"""Video e2e over a real server: preview fps, framing, latest-wins, MJPEG."""

from __future__ import annotations

import time

import httpx
import pytest
from apollo_xarm7_core.protocol import HEADER_SIZE, unpack_header
from conftest import LiveServer, make_runtime_config
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect as ws_connect

SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "single_rail",
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = LiveServer(make_runtime_config(tmp_path_factory.mktemp("rt")))
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=30.0) as client:
        yield client


def _recv_frame(sock, timeout=5.0) -> bytes:
    buf = sock.recv(timeout=timeout)
    assert isinstance(buf, bytes)
    return buf


def test_pre_session_preview_at_15fps_and_sim_closed_1008(server, api):
    assert api.get("/api/session").status_code == 404  # truly pre-session
    with ws_connect(f"{server.ws}/ws/video/cam_front") as sock:
        first = _recv_frame(sock)
        ts, n = unpack_header(first)
        assert n == len(first) - HEADER_SIZE
        assert first[HEADER_SIZE:HEADER_SIZE + 2] == b"\xff\xd8"  # JPEG SOI
        t0 = time.monotonic()
        count = 0
        while time.monotonic() - t0 < 1.5:
            _recv_frame(sock)
            count += 1
        fps = count / (time.monotonic() - t0)
        assert 11 <= fps <= 19, fps  # ~15 fps preview
    with ws_connect(f"{server.ws}/ws/video/sim") as sock:  # session-only
        with pytest.raises(ConnectionClosed) as exc:
            sock.recv(timeout=5)
        assert exc.value.rcvd.code == 1008


def test_session_sim_stream_framing_latest_wins_and_mjpeg(server, api):
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    try:
        with ws_connect(f"{server.ws}/ws/video/sim", max_size=2**23) as sock:
            buf = _recv_frame(sock, timeout=10)
            ts, n = unpack_header(buf)
            assert n == len(buf) - HEADER_SIZE
            assert abs(time.monotonic() - ts) < 5.0  # server-monotonic seconds
            assert buf[HEADER_SIZE:HEADER_SIZE + 2] == b"\xff\xd8"
            # Deliberately stall 2 s, then resume reading: the per-client
            # depth-1 latest slot lets the client catch back up to a LIVE frame
            # (near the resume time) far faster than real time, rather than
            # replaying 2 s of frames paced at the render fps.
            time.sleep(2.0)
            resume = time.monotonic()
            newest_ts = 0.0
            for _ in range(2000):
                frame = sock.recv(timeout=5)
                newest_ts = unpack_header(frame)[0]
                if newest_ts >= resume - 0.1:
                    break
            assert newest_ts >= resume - 0.1  # reached a live frame
            assert time.monotonic() - resume < 1.0  # ...in far less than 2 s
            # MJPEG debug endpoint shares the SAME encoded buffers.
            ws_payloads = set()
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:
                frame = sock.recv(timeout=2)
                ws_payloads.add(frame[HEADER_SIZE:])
            mjpeg_payloads = _collect_mjpeg(server, "sim", n_frames=5)
            assert mjpeg_payloads & ws_payloads  # byte-for-byte identical JPEGs
    finally:
        api.delete("/api/session")


def _collect_mjpeg(server, stream_id: str, n_frames: int) -> set[bytes]:
    payloads: set[bytes] = set()
    with httpx.Client(timeout=10.0) as client:
        with client.stream("GET", f"{server.http}/video/{stream_id}.mjpg") as resp:
            assert resp.status_code == 200
            assert "multipart/x-mixed-replace" in resp.headers["content-type"]
            body = b""
            for chunk in resp.iter_bytes():
                body += chunk
                while True:
                    start = body.find(b"\r\n\r\n")
                    if start < 0:
                        break
                    header = body[:start].decode(errors="replace")
                    length = None
                    for line in header.splitlines():
                        if line.lower().startswith("content-length:"):
                            length = int(line.split(":", 1)[1])
                    if length is None or len(body) < start + 4 + length:
                        break
                    payloads.add(body[start + 4:start + 4 + length])
                    body = body[start + 4 + length:]
                if len(payloads) >= n_frames:
                    return payloads
    return payloads


def test_unknown_mjpeg_stream_404(server, api):
    assert httpx.get(f"{server.http}/video/never.mjpg").status_code == 404
