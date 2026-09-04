"""WS/REST contract tests over create_app + a sim workcell (04-runtime §13)."""

from __future__ import annotations

import pytest
from conftest import make_runtime_config
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app

SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["arm0"],
    "frames": {"arm0": "arm_base:arm0"},
    "sim_scene": "single_rail",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cfg = make_runtime_config(tmp_path_factory.mktemp("rt"))
    app = create_app(Runtime(cfg))
    with TestClient(app) as c:
        yield c
        c.delete("/api/session")


@pytest.fixture()
def session(client):
    r = client.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    yield r.json()
    client.delete("/api/session")


def test_health_and_epoch(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and len(body["epoch"]) == 32


def test_keymap_matches_core(client):
    from apollo_mavis_v2_core.protocol import KEYMAP

    served = client.get("/api/keymap").json()
    assert served == [e.model_dump() for e in KEYMAP]
    assert len(served) == 23  # 13-tracker §3: + KeyC tracker_clutch, KeyZ switch_arm_prev
    by_code = {row["code"]: row for row in served}
    assert by_code["KeyC"]["gamepad"] == "RT" and by_code["KeyZ"]["gamepad"] == "LB"


def test_scenes_listing(client):
    scenes = client.get("/api/scenes", params={"kind": "sim"}).json()
    ids = {s["scene_id"] for s in scenes}
    assert "single_rail" in ids and "guardrail_env" in ids
    row = next(s for s in scenes if s["scene_id"] == "single_rail")
    assert row["num_arms"] == 1 and row["rail_flags"] == [True]
    assert client.get("/api/scenes", params={"kind": "nope"}).status_code == 422


def test_workcell_and_cameras_pre_session(client):
    ws = client.get("/api/workcell").json()
    assert ws["kind"] == "sim" and ws["available_kinds"] == ["sim"]
    arm = ws["arms"][0]
    assert arm["has_rail"] and len(arm["joint_limits"]) == 8
    assert arm["joint_limits"][7] == [0.0, 0.65]
    cams = client.get("/api/cameras").json()
    assert {c["camera_id"] for c in cams} == {"cam_front", "arm0_wrist_cam"}
    assert all(c["live"] for c in cams)


def test_session_404_then_lifecycle(client):
    assert client.get("/api/session").status_code == 404
    r = client.post("/api/session", json=SPEC)
    assert r.status_code == 200
    info = r.json()
    assert info["mode"] == "teleop" and "sim" in info["streams"]
    assert client.post("/api/session", json=SPEC).status_code == 409  # singleton
    got = client.get("/api/session").json()
    assert got["session_id"] == info["session_id"]
    assert client.delete("/api/session").status_code == 204
    assert client.delete("/api/session").status_code == 204  # idempotent
    assert client.get("/api/session").status_code == 404


def test_post_session_409_matrix(client):
    bad_kind = dict(SPEC, kind="hardware", digital_twin_scene="single_rail")
    assert client.post("/api/session", json=bad_kind).status_code == 409
    bad_mode = dict(SPEC, mode="dagger", task="t")  # collect lands in phase-07
    assert client.post("/api/session", json=bad_mode).status_code == 409
    bad_arms = dict(SPEC, arms=["arm9"], frames={})
    assert client.post("/api/session", json=bad_arms).status_code == 409
    bad_scene = dict(SPEC, sim_scene="not_a_scene")
    assert client.post("/api/session", json=bad_scene).status_code == 409
    bad_profile = dict(SPEC, start_from="profile:doesnotexist")
    assert client.post("/api/session", json=bad_profile).status_code == 409
    invalid = dict(SPEC, start_from="gohome")  # fails core regex -> 422
    assert client.post("/api/session", json=invalid).status_code == 422


def test_ws_control_hello_roles_and_observer_nack(client, session):
    with client.websocket_connect("/ws/control") as ws1:
        hello1 = ws1.receive_json()
        assert hello1["t"] == "hello" and hello1["role"] == "controller"
        assert hello1["epoch"] and hello1["session_id"] == session["session_id"]
        with client.websocket_connect("/ws/control") as ws2:
            hello2 = ws2.receive_json()
            assert hello2["role"] == "observer"  # never close-1008
            ws2.send_json({"t": "action", "name": "switch_arm"})
            ack = ws2.receive_json()
            assert ack == {"t": "ack", "name": "switch_arm", "ok": False,
                           "detail": "observer"}
        ws1.send_json({"t": "action", "name": "switch_arm"})
        ack = ws1.receive_json()
        assert ack["ok"] is True


def test_ws_control_no_session_nack(client):
    with client.websocket_connect("/ws/control") as ws:
        assert ws.receive_json()["session_id"] is None
        ws.send_json({"t": "action", "name": "switch_arm"})
        ack = ws.receive_json()
        assert not ack["ok"] and ack["detail"] == "no session"


def test_ws_control_invalid_args_nack(client, session):
    with client.websocket_connect("/ws/control") as ws:
        ws.receive_json()
        ws.send_json({"t": "action", "name": "joint_target", "args": {"nope": 1}})
        ack = ws.receive_json()
        assert not ack["ok"] and "invalid args" in ack["detail"]


def test_ws_video_unknown_and_reserved_pre_session(client):
    with client.websocket_connect("/ws/video/never-existed") as ws:
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_bytes()
        assert exc.value.code == 1008
    with client.websocket_connect("/ws/video/sim") as ws:  # session-only id
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_bytes()
        assert exc.value.code == 1008


def test_telemetry_pre_session_idle(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_json()
        assert msg["t"] == "telemetry"
        assert msg["session"]["state"] == "idle"
        assert msg["arms"] == [] and msg["collision"]["severity"] == "ok"
        # Tracker block is populated pre-session (device fields; backend none here).
        trk = msg["tracker"]
        assert trk["backend"] == "none" and trk["status"] == "no_backend"
        assert trk["clutch"] is False and trk["engaged_arm"] is None
        assert trk["settings"] == {
            "yaw_deg": 0.0, "pos_scale": 1.0, "follow_rotation": True,
            "filter_enabled": True, "filter_min_cutoff_hz": 1.0, "filter_beta": 0.05,
        }
        assert trk["pose_filtered"] is None and trk["device_action"] is None
