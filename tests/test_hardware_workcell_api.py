"""Phase-11 REST/WS contract over a config with BOTH workcells, a fake
microphone and fake hardware cameras (no hardware): ``GET /api/workcell?kind=``,
``GET /api/microphones``, ``GET /api/cameras`` with failure-isolated hardware
previews, and the telemetry ``microphone`` block."""

from __future__ import annotations

import socket
import time

import pytest
from apollo_mavis_v2_core import CameraInitError, WorkcellConfig
from apollo_mavis_v2_core.testing import FakeCamera
from conftest import AcceptingListener, LiveServer, make_runtime_config
from starlette.testclient import TestClient

from apollo_mavis_v2_runtime.config import (
    HardwareMonitorConfig,
    HardwareProbeConfig,
    MicrophoneConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app

SCENE = "mavis_v2"


def _closed_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


HW_WORKCELL = {
    "kind": "hardware",
    "digital_twin_scene": SCENE,
    "arms": [  # view listed first on purpose: the status rows must still lead with grip
        {"id": "view", "ip": "127.0.0.1", "base_in_world": {}, "gripper": "none",
         "microphone": True},
        {"id": "grip", "ip": "127.0.0.1", "base_in_world": {}, "gripper": "xarm_g2"},
    ],
    "cameras": [
        {"id": "camera1", "kind": "v4l2", "device_path": "/dev/v4l/by-id/TODO-camera1",
         "resolution": [64, 48], "fps": 30},
        {"id": "camera2", "kind": "v4l2", "device_path": "/dev/v4l/by-id/TODO-camera2",
         "resolution": [64, 48], "fps": 30},
    ],
    "safety": {"enabled": True},
}


def _config(tmp_path, *, probe_port: int, mic_backend: str = "fake"):
    cfg = make_runtime_config(tmp_path, SCENE)
    return cfg.model_copy(update={
        "workcells": {**cfg.workcells, "hardware": WorkcellConfig.model_validate(HW_WORKCELL)},
        "microphone": MicrophoneConfig(enabled=True, backend=mic_backend, label="RØDE NT-USB Mini"),
        "hardware_probe": HardwareProbeConfig(period_s=0.1, timeout_s=0.5, port=probe_port),
        # phase-09a: no read-only SDK client against 127.0.0.1 here and no overlay
        # (camera1/camera2 are not wrist cameras anyway); tests/test_twin_overlay.py
        # covers both with fakes.
        "hardware_monitor": HardwareMonitorConfig(enabled=False),
        "twin_overlay": TwinOverlayConfig(enabled=False),
    })


def _camera_factory(cam_cfg):
    """camera1 opens (FakeCamera); camera2 is absent (CameraInitError) — isolation."""
    if cam_cfg.id == "camera1":
        return FakeCamera(cam_cfg.id, tuple(cam_cfg.resolution), cam_cfg.fps)
    raise CameraInitError("camera", f"{cam_cfg.id}: cannot open {cam_cfg.device_path}")


def _wait(pred, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


@pytest.fixture(scope="module")
def probe_port():
    """Port of a fake "control box" that stays ``open`` for the whole module.

    The module-scoped ``Runtime`` below probes it at 10 Hz x 2 arms from the
    moment it is constructed; a listener nobody accepts on would saturate its
    accept queue after ~9 connections (~0.5 s) and flip both arms to
    ``unreachable``, so the listener drains connections on a daemon thread.
    """
    listener = AcceptingListener()
    yield listener.port
    listener.close()


@pytest.fixture(scope="module")
def rt(tmp_path_factory, probe_port):
    cfg = _config(tmp_path_factory.mktemp("rt"), probe_port=probe_port)
    runtime = Runtime(cfg)
    runtime.manager.camera_factory = _camera_factory
    return runtime


@pytest.fixture(scope="module")
def client(rt):
    app = create_app(rt)
    with TestClient(app) as c:
        yield c
        c.delete("/api/session")


def test_workcell_default_and_kinds(client, rt):
    ws = client.get("/api/workcell").json()
    assert ws["kind"] == "sim" and ws["available_kinds"] == ["hardware", "sim"]
    # sim preview scene rows; the Manipulation Arm leads although the scene order is view, grip
    assert [a["arm_id"] for a in ws["arms"]] == ["grip", "view"]
    assert all(a["reachable"] == "unknown" and a["ip"] is None for a in ws["arms"])
    # default lists everything previewed: 4 sim cameras + 2 hardware cameras
    ids = {c["camera_id"] for c in ws["cameras"]}
    assert {"cam_front", "cam_top", "view_wrist_cam", "grip_wrist_cam", "camera1", "camera2"} <= ids

    sim = client.get("/api/workcell", params={"kind": "sim"}).json()
    assert sim["kind"] == "sim" and sim["arms"] == ws["arms"]
    assert {c["camera_id"] for c in sim["cameras"]} == {
        "cam_front", "cam_top", "view_wrist_cam", "grip_wrist_cam"
    }
    assert all(c["live"] and c["kind"] == "sim" for c in sim["cameras"])

    assert _wait(lambda: rt.hardware_probe.rounds >= 1)
    hw = client.get("/api/workcell", params={"kind": "hardware"}).json()
    assert hw["kind"] == "hardware" and hw["available_kinds"] == ["hardware", "sim"]
    by_id = {a["arm_id"]: a for a in hw["arms"]}
    assert [a["arm_id"] for a in hw["arms"]] == ["grip", "view"]  # config order was view, grip
    for a in by_id.values():
        assert a["ip"] == "127.0.0.1" and a["connected"] is False  # no session
        assert a["reachable"] == "open"  # local listening socket on the probe port
        assert a["has_rail"] is True and len(a["joint_limits"]) == 8  # from the twin scene
        assert a["joint_limits"][7] == [0.0, 0.65]
    assert by_id["grip"]["gripper"] == "xarm_g2" and by_id["view"]["gripper"] == "none"
    assert by_id["grip"]["gripper_force_capable"] and not by_id["view"]["gripper_force_capable"]
    assert hw["hardware_ready"] is True and ws["hardware_ready"] is True
    assert all(a["error_code"] == 0 for a in hw["arms"])  # monitor disabled -> 0
    cams = {c["camera_id"]: c for c in hw["cameras"]}
    assert set(cams) == {"camera1", "camera2"}  # no *_align rows: overlay disabled
    assert cams["camera1"]["live"] is True and cams["camera1"]["kind"] == "v4l2"
    assert cams["camera2"]["live"] is False  # absent -> black tile, no WS
    assert cams["camera2"]["resolution"] == [64, 48] and cams["camera2"]["fps"] == 30
    assert client.get("/api/workcell", params={"kind": "twin"}).status_code == 422


def test_cameras_lists_sim_previews_and_hardware_with_isolation(client, rt):
    cams = {c["camera_id"]: c for c in client.get("/api/cameras").json()}
    assert cams["camera1"]["live"] is True and cams["camera2"]["live"] is False
    assert cams["cam_front"]["live"] is True and cams["cam_front"]["kind"] == "sim"
    assert rt.hub.has("camera1") and not rt.hub.has("camera2")
    assert "camera2" in rt.manager.hardware_camera_errors()
    assert "CameraInitError" in rt.manager.hardware_camera_errors()["camera2"]
    # The live hardware preview streams real frames (FakeCamera gradient) over WS.
    with client.websocket_connect("/ws/video/camera1") as ws:
        first = ws.receive_bytes()
        assert len(first) > 12 and first[12:14] == b"\xff\xd8"  # <dI header + JPEG SOI


def test_microphones_and_telemetry_block_fake(client, rt):
    assert _wait(lambda: rt.microphone.status().status == "live")
    mics = client.get("/api/microphones").json()
    assert len(mics) == 1
    mic = mics[0]
    assert mic["mic_id"] == "mic_view" and mic["label"] == "RØDE NT-USB Mini"
    assert mic["kind"] == "fake" and mic["source"] is None and "source" in mic
    assert mic["sample_rate"] == 48000 and mic["channels"] == 1
    assert mic["live"] is True and mic["status"] == "live" and mic["detail"] == ""
    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_json()
        assert msg["session"]["state"] == "idle"
        hm = msg["hardware_monitor"]  # phase-09a: inert block, one row per hardware arm
        assert hm["enabled"] is False and hm["paused"] is False and hm["overlays"] == []
        assert [a["arm_id"] for a in hm["arms"]] == ["view", "grip"]  # config order
        assert all(a["status"] == "off" and "disabled" in a["detail"] for a in hm["arms"])
        block = msg["microphone"]
        assert block is not None and block["mic_id"] == "mic_view"
        assert block["status"] == "live" and block["seq"] > 0
        assert len(block["env_min"]) == 64 and len(block["env_max"]) == 64
        assert all(isinstance(v, int) and -127 <= v <= 127 for v in block["env_min"])
        assert block["rms_dbfs"] < block["peak_dbfs"] <= 0.0 and block["clipping"] is False
        assert block["sample_rate"] == 48000 and block["age_s"] is not None
        # Frames are aligned to telemetry_hz: consecutive frames carry new seqs.
        seqs = {block["seq"]}
        for _ in range(10):
            seqs.add(ws.receive_json()["microphone"]["seq"])
        assert len(seqs) >= 5


def test_hardware_previews_survive_a_sim_session_and_sim_previews_stop(client, rt):
    spec = {
        "mode": "teleop", "kind": "sim", "arms": ["grip"],
        "frames": {"grip": "arm_base:grip"}, "sim_scene": SCENE,
    }
    r = client.post("/api/session", json=spec)
    assert r.status_code == 200, r.text
    try:
        assert rt.hub.has("camera1") and rt.hub.has("sim")  # hardware preview kept
        cams = {c["camera_id"]: c for c in client.get("/api/cameras").json()}
        assert cams["camera1"]["live"] is True and cams["camera2"]["live"] is False
        assert client.get("/api/microphones").json()[0]["live"] is True
        hw = client.get("/api/workcell", params={"kind": "hardware"}).json()
        assert all(a["connected"] is False for a in hw["arms"])  # sim session != hw session
        assert client.get("/api/workcell").json()["kind"] == "sim"
    finally:
        assert client.delete("/api/session").status_code == 204
    assert _wait(lambda: rt.hub.has("cam_front"))  # sim previews back
    assert rt.hub.has("camera1")


def test_hardware_session_still_409_until_phase_09(client):
    spec = {
        "mode": "teleop", "kind": "hardware", "arms": ["grip"],
        "frames": {"grip": "arm_base:grip"}, "digital_twin_scene": SCENE,
    }
    r = client.post("/api/session", json=spec)
    assert r.status_code == 409 and "phase-09" in r.json()["detail"]


def test_hardware_ready_false_when_boxes_refuse(tmp_path):
    """Closed port == control box booting: reachable 'refused', not ready."""
    cfg = _config(tmp_path, probe_port=_closed_port(), mic_backend="none")
    runtime = Runtime(cfg)
    runtime.manager.camera_factory = _camera_factory
    with TestClient(create_app(runtime)) as c:
        assert _wait(lambda: runtime.hardware_probe.rounds >= 1)
        hw = c.get("/api/workcell", params={"kind": "hardware"}).json()
        assert {a["reachable"] for a in hw["arms"]} == {"refused"}
        assert hw["hardware_ready"] is False
        mics = c.get("/api/microphones").json()  # backend none: listed, not live
        assert len(mics) == 1 and mics[0]["kind"] == "none" and mics[0]["live"] is False
        assert mics[0]["status"] == "no_backend" and "none" in mics[0]["detail"]
        tele = c.websocket_connect("/ws/telemetry")
        with tele as ws:
            block = ws.receive_json()["microphone"]
            assert block["status"] == "no_backend" and block["env_min"] == []


def test_hardware_camera_factory_import_failure_isolated(tmp_path, monkeypatch):
    """No usable camera factory (e.g. hardware extra missing): every hardware
    camera is live:false, sim previews untouched, nothing raises."""
    cfg = _config(tmp_path, probe_port=_closed_port(), mic_backend="none")
    runtime = Runtime(cfg)

    def broken_factory(self):
        raise ImportError("No module named 'apollo_mavis_v2_hardware'")

    monkeypatch.setattr(type(runtime.manager), "_default_camera_factory", broken_factory)
    with TestClient(create_app(runtime)) as c:
        cams = {x["camera_id"]: x for x in c.get("/api/cameras").json()}
        assert cams["camera1"]["live"] is False and cams["camera2"]["live"] is False
        assert cams["cam_front"]["live"] is True
        errs = runtime.manager.hardware_camera_errors()
        assert set(errs) == {"camera1", "camera2"} and "ImportError" in errs["camera1"]


def test_live_server_telemetry_microphone_over_real_websocket(tmp_path, probe_port):
    """Real uvicorn: the block rides the 25 Hz broadcast with distinct seqs."""
    import json

    from websockets.sync.client import connect as ws_connect

    cfg = _config(tmp_path, probe_port=probe_port)
    server = LiveServer(cfg)
    try:
        assert _wait(lambda: server.runtime.microphone.status().status == "live")
        with ws_connect(f"{server.ws}/ws/telemetry") as sock:
            frames = [json.loads(sock.recv(timeout=5)) for _ in range(12)]
        blocks = [f["microphone"] for f in frames]
        assert all(b["status"] == "live" for b in blocks)
        seqs = [b["seq"] for b in blocks]
        assert seqs == sorted(seqs) and len(set(seqs)) >= 6
        assert all(len(b["env_max"]) == 64 for b in blocks)
    finally:
        server.stop()
