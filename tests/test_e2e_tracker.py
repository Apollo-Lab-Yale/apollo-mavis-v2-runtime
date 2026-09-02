"""13-tracker §4 e2e: the ``fake`` tracker backend on ``mavis_v2`` through a
REAL uvicorn server — clutch via ``KeyC`` moves the EE, telemetry carries
``tracker.*`` pre-session and in-session, settings/switch_arm_prev ack, no
ERROR logs."""

from __future__ import annotations

import json
import logging
import math
import time

import httpx
import pytest
from conftest import LiveServer, make_runtime_config
from websockets.sync.client import connect as ws_connect

from apollo_xarm7_runtime.config import TrackerConfig

SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["view", "grip"],
    "frames": {"view": "arm_base:view", "grip": "arm_base:grip"},
    "sim_scene": "mavis_v2",
}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    cfg = make_runtime_config(
        tmp_path_factory.mktemp("rt"), scene="mavis_v2", tracker=TrackerConfig(backend="fake")
    )
    srv = LiveServer(cfg)
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def api(server):
    with httpx.Client(base_url=server.http, timeout=30.0) as client:
        yield client


class Ctl:
    def __init__(self, server):
        self.sock = ws_connect(f"{server.ws}/ws/control")
        self.hello = json.loads(self.sock.recv(timeout=5))
        self.seq = 0

    def keys(self, held: list[str]) -> None:
        self.seq += 1
        self.sock.send(json.dumps({"t": "keys", "seq": self.seq, "ts": time.time(), "held": held}))

    def action(self, name: str, args: dict | None = None) -> dict:
        self.sock.send(json.dumps({"t": "action", "name": name, "args": args or {}}))
        ack = json.loads(self.sock.recv(timeout=10))
        assert ack["t"] == "ack" and ack["name"] == name, ack
        return ack

    def close(self) -> None:
        self.sock.close()


class Tele:
    def __init__(self, server):
        self.sock = ws_connect(f"{server.ws}/ws/telemetry")

    def latest(self) -> dict:
        msg = json.loads(self.sock.recv(timeout=5))
        while True:
            try:
                msg = json.loads(self.sock.recv(timeout=0.001))
            except TimeoutError:
                return msg

    def close(self) -> None:
        self.sock.close()


def _dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


@pytest.fixture()
def session(api):
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    for _ in range(200):
        if api.get("/api/session").json()["state"] == "running":
            break
        time.sleep(0.05)
    yield r.json()
    api.delete("/api/session")


def test_tracker_telemetry_pre_session_device_fields(server, api):
    assert api.get("/api/session").status_code == 404
    tele = Tele(server)
    try:
        deadline = time.monotonic() + 5.0
        trk = tele.latest()["tracker"]
        while trk["status"] != "tracking" and time.monotonic() < deadline:
            trk = tele.latest()["tracker"]
        assert trk["backend"] == "fake" and trk["status"] == "tracking", trk
        assert trk["object_name"] == "WM0" and trk["seq"] > 0 and trk["rate_hz"] > 50
        assert trk["age_s"] is not None and trk["age_s"] < 0.2
        assert trk["pose_raw"] is not None and trk["pose_world"] is not None
        assert _dist(trk["pose_raw"]["position"], trk["pose_world"]["position"]) < 1e-9  # yaw 0
        assert trk["clutch"] is False and trk["engaged_arm"] is None
        assert trk["anchor_tcp"] is None and trk["target_tcp"] is None
        assert trk["settings"] == {"yaw_deg": 0.0, "pos_scale": 1.0, "follow_rotation": True}
    finally:
        tele.close()


def test_clutch_keyc_moves_ee_then_release_holds(server, api, session, caplog):
    caplog.set_level(logging.INFO)
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        assert ctl.hello["role"] == "controller"
        msg = tele.latest()
        assert msg["active_arm"] == "view"
        p0 = msg["arms"][0]["ee_pose"]["position"]
        engaged_frames = 0
        end = time.monotonic() + 2.0
        while time.monotonic() < end:  # KeysMsg + 25 Hz heartbeat with the clutch held
            ctl.keys(["KeyC"])
            time.sleep(0.04)
            trk = tele.latest()["tracker"]
            if trk["engaged_arm"] == "view":
                engaged_frames += 1
                assert trk["clutch"] is True
                assert trk["anchor_tcp"] is not None and trk["target_tcp"] is not None
                assert trk["status"] == "tracking"
        assert engaged_frames > 20, engaged_frames
        msg = tele.latest()
        p1 = msg["arms"][0]["ee_pose"]["position"]
        assert _dist(p0, p1) > 0.03, (p0, p1)  # fake circle ~0.047 m/s for 2 s
        trk = msg["tracker"]
        # The target never leads the anchored hand by more than the hand moved.
        assert _dist(trk["target_tcp"]["position"], trk["anchor_tcp"]["position"]) > 0.01
        ctl.keys([])  # release: hold-last, anchors cleared
        time.sleep(0.4)
        msg = tele.latest()
        trk = msg["tracker"]
        assert trk["clutch"] is False and trk["engaged_arm"] is None
        assert trk["anchor_tcp"] is None and trk["target_tcp"] is None
        p2 = msg["arms"][0]["ee_pose"]["position"]
        time.sleep(0.5)
        p3 = tele.latest()["arms"][0]["ee_pose"]["position"]
        assert _dist(p2, p3) < 2e-3  # stopped while the fake tracker keeps circling
        # The non-active arm never moved.
        assert _dist(msg["arms"][1]["ee_pose"]["position"],
                     tele.latest()["arms"][1]["ee_pose"]["position"]) < 2e-3
    finally:
        ctl.close()
        tele.close()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], [r.getMessage() for r in errors]


def test_tracker_settings_and_switch_arm_prev_over_ws(server, api, session):
    ctl = Ctl(server)
    tele = Tele(server)
    try:
        ack = ctl.action("tracker_settings", {"pos_scale": 2.0, "yaw_deg": 30.0})
        assert ack["ok"], ack
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            s = tele.latest()["tracker"]["settings"]
            if s["pos_scale"] == 2.0:
                break
        assert s == {"yaw_deg": 30.0, "pos_scale": 2.0, "follow_rotation": True}
        trk = tele.latest()["tracker"]  # pose_world now yaw-rotated vs pose_raw
        assert _dist(trk["pose_raw"]["position"], trk["pose_world"]["position"]) > 0.01
        ack = ctl.action("tracker_settings", {"pos_scale": 9.0})
        assert not ack["ok"] and "invalid args" in ack["detail"]
        ack = ctl.action("tracker_settings", {"pos_scale": 1.0, "yaw_deg": 0.0})
        assert ack["ok"]
        ack = ctl.action("switch_arm_prev")
        assert ack["ok"] and ack["detail"] == "grip"  # wraps from view
        ack = ctl.action("switch_arm_prev")
        assert ack["ok"] and ack["detail"] == "view"
        ack = ctl.action("switch_arm_prev", {"nope": 1})
        assert not ack["ok"] and "invalid args" in ack["detail"]
    finally:
        ctl.close()
        tele.close()
