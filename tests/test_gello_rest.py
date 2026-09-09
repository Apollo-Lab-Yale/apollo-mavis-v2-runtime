"""REST + telemetry contract of the GELLO leader (phase-15; 16-gello §9.2 / D10) over
create_app: ``GET /api/gello`` shape, ``POST /api/gello/calibrate`` (match_arm in sim and
against the hardware monitor's sample, the gripper endpoints, clear, the 409 matrix), and
the ``telemetry.gello`` block. Style of tests/test_server_contract.py."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from apollo_mavis_v2_core import WorkcellConfig
from apollo_mavis_v2_core.protocol import GelloCalibrateResult, GelloInfo, GelloTelemetry
from conftest import FakeMonitorFactory, FakeMonitorSample, make_runtime_config
from starlette.testclient import TestClient
from test_gello_device import FakeBus, fake_sdk

from apollo_mavis_v2_runtime.config import (
    GELLO_VIEW_POSTURE_RAD,
    GelloConfig,
    HardwareMonitorConfig,
    HardwareProbeConfig,
    MicrophoneConfig,
    TwinOverlayConfig,
)
from apollo_mavis_v2_runtime.devices.gello import FAKE_DEFAULT_Q, GelloReader
from apollo_mavis_v2_runtime.runtime import Runtime
from apollo_mavis_v2_runtime.server.app import create_app

PI = math.pi
MAVIS_SIM_WORKCELL = {
    "kind": "sim",
    "sim_scene": "mavis_v2",
    "arms": [
        {"id": "view", "base_in_world": {}, "gripper": "none"},
        {"id": "grip", "base_in_world": {}},
    ],
    "cameras": [],
}
TELEOP_SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["view", "grip"],
    "frames": {"view": "arm_base:view", "grip": "arm_base:grip"},
    "sim_scene": "mavis_v2",
}
HW_WORKCELL = {
    "kind": "hardware",
    "digital_twin_scene": "mavis_v2",
    "arms": [
        {"id": "grip", "ip": "192.168.1.201", "base_in_world": {}, "gripper": "xarm_g2"},
        {"id": "view", "ip": "192.168.2.219", "base_in_world": {}, "gripper": "none"},
    ],
    "cameras": [],
    "safety": {"enabled": True},
}
GRIP_MONITOR_Q = (PI, 0.0, 0.5, 0.3, 0.0, -0.2, 0.0)


def _wait(pred, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _config(tmp_path, *, hardware: bool = False):
    cfg = make_runtime_config(tmp_path, "mavis_v2")
    update = {
        "workcells": {"sim": WorkcellConfig.model_validate(MAVIS_SIM_WORKCELL)},
        "gello": GelloConfig(backend="fake", calibration_path=tmp_path / "gello_calibration.json"),
        "microphone": MicrophoneConfig(enabled=False),
        "hardware_probe": HardwareProbeConfig(enabled=False),
        "twin_overlay": TwinOverlayConfig(enabled=False),
    }
    if hardware:
        update["workcells"]["hardware"] = WorkcellConfig.model_validate(HW_WORKCELL)
        update["hardware_monitor"] = HardwareMonitorConfig()
    else:
        update["hardware_monitor"] = HardwareMonitorConfig(enabled=False)
    return cfg.model_copy(update=update)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rt = Runtime(_config(tmp_path_factory.mktemp("rt")))
    with TestClient(create_app(rt)) as c:
        assert _wait(lambda: rt.gello.status().status == "connected")
        assert _wait(lambda: "grip" in rt.manager.idle_sim_q(), 20.0)  # previews up
        yield c
        c.delete("/api/session")


def _leader(client) -> np.ndarray:
    return np.asarray(client.get("/api/gello").json()["q_raw"])


def test_get_gello_shape(client):
    r = client.get("/api/gello")
    assert r.status_code == 200, r.text
    info = GelloInfo.model_validate(r.json())
    assert info.backend == "fake" and info.status == "connected" and info.port == "fake"
    assert info.baud is None and info.seq > 0 and info.rate_hz > 50 and info.age_s is not None
    assert np.allclose(info.q_raw, FAKE_DEFAULT_Q) and np.allclose(info.q, FAKE_DEFAULT_Q)
    assert info.gripper_frac == 1.0 and info.calibrated and info.joint_offsets_rad == [0.0] * 7
    assert info.joint_signs == [1] * 7
    assert info.scene_id == "mavis_v2_kitchen"
    assert info.scene_label == "APOLLO MAVIS V2 Kitchen (GELLO)"
    assert info.view_posture_rad == list(GELLO_VIEW_POSTURE_RAD) and info.view_rail_m == 0.0
    assert info.calibration_path.endswith("gello_calibration.json")
    assert info.hardware_admitted is True  # 16-gello D8
    assert info.gripper_open_rad is None and info.gripper_closed_rad is None
    # the kitchen stays hidden from the listing (16-gello D6): only mavis_v2 is exposed
    assert {s["scene_id"] for s in client.get("/api/scenes").json()} == {"mavis_v2"}


def test_calibrate_match_arm_in_sim_writes_the_file_and_the_ops_are_echoed(client):
    rt = client.app.state.runtime
    path = rt.gello_calibration.path
    q_arm = rt.manager.idle_sim_q()["grip"][:7]
    assert np.allclose(q_arm, FAKE_DEFAULT_Q)  # the mavis_v2 keyframe: parked posture
    # pose the (fake) leader like the arm plus a quarter-turn horn offset and a bit of error
    rt.gello.fake_set(q_arm + np.array([PI / 2, 0.0, -PI, 0.0, 0.0, 0.0, 0.08]), 1.0)
    r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "sim"})
    assert r.status_code == 200, r.text
    res = GelloCalibrateResult.model_validate(r.json())
    assert res.ok and np.allclose(res.joint_offsets_rad, [PI / 2, 0, -PI, 0, 0, 0, 0])
    assert "residual" in res.detail and "0.080" in res.detail
    assert path.is_file()
    assert np.allclose(rt.gello_calibration.load().joint_offsets_rad, res.joint_offsets_rad)
    # the gripper endpoints: raw reading at the two poses
    r = client.post("/api/gello/calibrate", json={"op": "gripper_open", "kind": "sim"})
    assert r.status_code == 200 and r.json()["gripper_open_rad"] == 1.0
    rt.gello.fake_set(q_arm, 0.0)
    r = client.post("/api/gello/calibrate", json={"op": "gripper_closed", "kind": "sim"})
    assert r.status_code == 200, r.text
    assert (r.json()["gripper_open_rad"], r.json()["gripper_closed_rad"]) == (1.0, 0.0)
    assert np.allclose(r.json()["joint_offsets_rad"], res.joint_offsets_rad)  # kept
    info = client.get("/api/gello").json()
    assert (info["gripper_open_rad"], info["gripper_closed_rad"]) == (1.0, 0.0)  # echoed
    assert info["joint_offsets_rad"] == [0.0] * 7  # the fake maps with identity (16-gello §4)
    # clear deletes the file
    r = client.post("/api/gello/calibrate", json={"op": "clear", "kind": "sim"})
    assert r.status_code == 200 and r.json()["ok"] and "cleared" in r.json()["detail"]
    assert not path.exists()
    info = client.get("/api/gello").json()
    assert info["gripper_open_rad"] is None and info["gripper_closed_rad"] is None
    r = client.post("/api/gello/calibrate", json={"op": "clear", "kind": "sim"})
    assert r.status_code == 200 and "no calibration file" in r.json()["detail"]


def test_calibrate_409_while_a_session_exists(client):
    r = client.post("/api/session", json=TELEOP_SPEC)
    assert r.status_code == 200, r.text
    try:
        for op in ("match_arm", "gripper_open", "clear"):
            r = client.post("/api/gello/calibrate", json={"op": op, "kind": "sim"})
            assert r.status_code == 409 and "end the session first" in r.json()["detail"], r.text
        assert client.get("/api/gello").status_code == 200  # never 409
    finally:
        client.delete("/api/session")


def test_calibrate_409_without_a_fresh_leader_sample(client):
    rt = client.app.state.runtime
    assert rt.gello.stop()
    try:
        assert _wait(lambda: rt.gello.status().status == "stale", 2.0)
        r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "sim"})
        assert r.status_code == 409 and "no fresh GELLO leader sample" in r.json()["detail"]
        assert "stale" in r.json()["detail"]
        r = client.post("/api/gello/calibrate", json={"op": "clear", "kind": "sim"})
        assert r.status_code == 200  # clear needs no leader
    finally:
        rt.gello.start()
        assert _wait(lambda: rt.gello.status().status == "connected")


def test_calibrate_409_when_the_kind_has_no_arm_posture(client):
    r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "hardware"})
    assert r.status_code == 409 and "no hardware workcell" in r.json()["detail"], r.text


def test_calibrate_422_on_a_malformed_body(client):
    def post(body):
        return client.post("/api/gello/calibrate", json=body).status_code

    assert post({"op": "nope", "kind": "sim"}) == 422
    assert post({"op": "match_arm"}) == 422
    assert post({"op": "match_arm", "kind": "x"}) == 422


def test_telemetry_carries_the_gello_device_half_without_a_session(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_json()
    block = msg["gello"]
    g = GelloTelemetry.model_validate(block)
    assert g.backend == "fake" and g.status == "connected" and g.calibrated
    assert g.state is None and g.state_detail == "" and g.lag_rad is None
    assert g.max_lag_rad is None and g.engaged_arm is None and g.viewpoint is None
    assert g.paused_latched is None  # 2026-09-09 review: the latch rides the session half
    assert list(block)[-1] == "paused_latched" and "gello" in msg  # appended last
    assert list(msg)[-1] == "gello"


def test_match_arm_accepts_an_uncalibrated_real_leader_and_the_next_sample_is_valid(client):
    """2026-09-09 review: ``match_arm`` - the op that CREATES the calibration - was refused
    on every uncalibrated dynamixel leader ("no fresh GELLO leader sample ... uncalibrated:
    run match_arm"), a chicken-and-egg 409. The fake dynamixel SDK stands in for the bus."""
    rt = client.app.state.runtime
    fake_reader = rt.gello
    path = rt.gello_calibration.path
    assert not path.exists()
    bus = FakeBus(baud=1_000_000)
    bus.ticks.update({1: 2048 + 1024})  # J1 raw = 3pi/2: a quarter-turn horn offset vs the arm
    real = GelloReader(
        GelloConfig(backend="dynamixel", baud=1_000_000, calibration_path=path),
        rt.bus.gello,
        import_dynamixel=lambda: fake_sdk(bus),
        calibration_store=rt.gello_calibration,
    )
    assert fake_reader.stop()
    rt.gello = real
    real.start()
    try:
        assert _wait(lambda: real.status().status == "connected")
        info = client.get("/api/gello").json()
        assert info["backend"] == "dynamixel" and not info["calibrated"]
        assert "uncalibrated" in info["detail"] and real.latest().valid is False
        r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "sim"})
        assert r.status_code == 200, r.text
        res = GelloCalibrateResult.model_validate(r.json())
        assert res.ok and np.allclose(res.joint_offsets_rad, [PI / 2, 0, 0, 0, 0, 0, 0])
        assert path.is_file()
        seq = real.latest().seq
        assert _wait(lambda: real.latest().seq > seq + 2 and real.latest().valid)
        st = real.status()
        assert st.calibrated and st.calibration_source == "file" and st.detail == ""
        assert np.allclose(real.latest().q, FAKE_DEFAULT_Q)  # sign * (raw - offset) = the arm
        assert real.fresh_sample() is not None  # the strict default is satisfied now
        # the gripper endpoint ops take the same path
        r = client.post("/api/gello/calibrate", json={"op": "gripper_open", "kind": "sim"})
        assert r.status_code == 200 and r.json()["gripper_open_rad"] == 0.0
        # a stopped (stale) leader still 409s, calibrated or not
        assert real.stop()
        assert _wait(lambda: real.status().status == "stale", 2.0)
        r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "sim"})
        assert r.status_code == 409 and "no fresh GELLO leader sample" in r.json()["detail"]
    finally:
        real.stop()
        rt.gello_calibration.clear()
        rt.gello = fake_reader
        fake_reader.reload_calibration()
        fake_reader.start()
        assert _wait(lambda: fake_reader.status().status == "connected")


def test_none_backend_runtime_reports_no_backend_and_refuses_calibration(tmp_path):
    cfg = make_runtime_config(tmp_path)  # single_rail, gello backend none (the default)
    rt = Runtime(cfg)
    assert rt.gello._thread is None
    with TestClient(create_app(rt)) as c:
        info = GelloInfo.model_validate(c.get("/api/gello").json())
        assert info.backend == "none" and info.status == "no_backend" and not info.calibrated
        assert info.q is None and info.q_raw is None and info.joint_offsets_rad is None
        assert info.scene_id == "mavis_v2_kitchen" and info.hardware_admitted
        r = c.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "sim"})
        assert r.status_code == 409 and "no_backend" in r.json()["detail"], r.text
        with c.websocket_connect("/ws/telemetry") as ws:
            assert ws.receive_json()["gello"]["status"] == "no_backend"


# -- hardware kind: the read-only monitor's sample is the arm posture ----------------------------
@pytest.fixture(scope="module")
def hw_client(tmp_path_factory):
    factory = FakeMonitorFactory(
        {"grip": FakeMonitorSample("grip", t_mono=time.monotonic(), q=GRIP_MONITOR_Q)}
    )  # no sample for the Perception Arm (irrelevant to match_arm)
    rt = Runtime(_config(tmp_path_factory.mktemp("hw"), hardware=True), monitor_factory=factory)
    with TestClient(create_app(rt)) as c:
        assert _wait(lambda: rt.hardware_monitor.status_of("grip")[0] == "running")
        assert _wait(lambda: rt.gello.status().status == "connected")
        yield c, factory
        c.delete("/api/session")


def _fresh_grip_sample(factory, q=GRIP_MONITOR_Q) -> None:
    factory.monitors["grip"].sample = FakeMonitorSample("grip", t_mono=time.monotonic(), q=q)


def test_calibrate_match_arm_hardware_uses_the_monitor_sample(hw_client):
    client, factory = hw_client
    rt = client.app.state.runtime
    _fresh_grip_sample(factory)  # the fixture's sample has aged past hardware_monitor.stale_s
    rt.gello.fake_set(np.asarray(GRIP_MONITOR_Q) + np.array([0.0, PI / 2, 0.0, 0.0, 0.0, 0.0, 0.0]))
    r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "hardware"})
    assert r.status_code == 200, r.text
    assert np.allclose(r.json()["joint_offsets_rad"], [0, PI / 2, 0, 0, 0, 0, 0])
    assert "hardware Manipulation Arm" in r.json()["detail"]
    # no monitor sample of the Manipulation Arm -> 409 with the monitor's status
    grip = factory.monitors["grip"]
    grip.sample = None
    try:
        r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "hardware"})
        assert r.status_code == 409 and "no fresh monitor sample" in r.json()["detail"], r.text
    finally:
        _fresh_grip_sample(factory)
    r = client.post("/api/gello/calibrate", json={"op": "clear", "kind": "hardware"})
    assert r.status_code == 200


def test_hardware_preview_and_match_arm_refuse_a_stale_or_disconnected_monitor_sample(hw_client):
    """2026-09-09 review: the hardware ``ArmStateMonitor`` never clears its last sample, so
    after a lost box / a paused monitor the preview kept evaluating an OUTDATED posture and
    said ``clear`` while ``POST /api/session`` 409'd. Now both the preview and ``match_arm``
    require a connected monitor status AND a sample younger than ``hardware_monitor.stale_s``;
    the preview reports ``no_workcell`` otherwise."""
    client, factory = hw_client
    rt = client.app.state.runtime
    rt.gello_preview.render_enabled = False  # the verdict is what matters here
    grip = factory.monitors["grip"]
    rt.gello.fake_set(np.asarray(FAKE_DEFAULT_Q))
    assert _wait(lambda: np.allclose(rt.gello.latest().q, FAKE_DEFAULT_Q))
    try:
        # warm the hardware kitchen twin first (its ~0.5 s build would age the STATIC fake
        # sample past stale_s; the real monitor refreshes at 10 Hz)
        assert client.post("/api/gello/preview", json={"kind": "hardware"}).status_code == 200
        # a fresh sample from a running monitor: evaluated (unwrapped against the arm)
        _fresh_grip_sample(factory)
        r = client.post("/api/gello/preview", json={"kind": "hardware"})
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["status"] in ("clear", "collision"), res
        assert res["q_goal"]["grip"][:7] == pytest.approx(list(FAKE_DEFAULT_Q))
        # the monitor lost the box (status error) but keeps the sample: NOT clear any more
        grip.forced_status = "error"
        grip.detail_text = "connection lost"
        r = client.post("/api/gello/preview", json={"kind": "hardware"})
        assert r.status_code == 200, r.text
        res = r.json()
        assert res["status"] == "no_workcell" and not res["ok"] and res["image_png_b64"] is None
        assert res["detail"] == (
            "no fresh monitor sample of the Manipulation Arm (monitor error: connection lost)"
        )
        r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "hardware"})
        assert r.status_code == 409, r.text
        assert r.json()["detail"].startswith(
            "no fresh monitor sample of the Manipulation Arm (monitor error: connection lost)"
        )
        # a paused monitor (released connections) is not connected either
        grip.forced_status = "paused"
        grip.detail_text = ""
        res = client.post("/api/gello/preview", json={"kind": "hardware"}).json()
        assert res["status"] == "no_workcell" and "monitor paused" in res["detail"]
        grip.forced_status = None
        # a running monitor whose sample is OLD (the box stopped answering): refused too
        grip.sample = FakeMonitorSample("grip", t_mono=time.monotonic() - 5.0, q=GRIP_MONITOR_Q)
        res = client.post("/api/gello/preview", json={"kind": "hardware"}).json()
        assert res["status"] == "no_workcell" and "s old" in res["detail"]
        r = client.post("/api/gello/calibrate", json={"op": "match_arm", "kind": "hardware"})
        assert r.status_code == 409 and "s old" in r.json()["detail"]
        # fresh again: evaluated again
        _fresh_grip_sample(factory)
        assert client.post("/api/gello/preview", json={"kind": "hardware"}).json()["status"] != (
            "no_workcell"
        )
    finally:
        grip.forced_status = None
        grip.detail_text = ""
        _fresh_grip_sample(factory)
        rt.gello_preview.render_enabled = True


def test_twin_overlay_scene_knob_overrides_the_workcell_twin_scene(tmp_path):
    """16-gello D6 / §10: ``twin_overlay.scene`` (null by default) picks the twin the
    ``*_align`` overlays render; the lab render sets ``mavis_v2_kitchen`` so the appliance
    outlines can be checked session-less. Only the overlay follows it - the gate / sweep
    twins keep the workcell's ``digital_twin_scene``."""
    base = _config(tmp_path, hardware=True).model_copy(
        update={"hardware_monitor": HardwareMonitorConfig(enabled=False)}
    )
    rt = Runtime(base)
    try:
        assert rt.twin_overlay is not None and rt.twin_overlay.twin_scene == "mavis_v2"
        assert rt.rail_sweep is not None and rt.rail_sweep.scene_id == "mavis_v2"
    finally:
        rt.gello.stop()
    kitchen = base.model_copy(
        update={"twin_overlay": TwinOverlayConfig(enabled=False, scene="mavis_v2_kitchen")}
    )
    rt = Runtime(kitchen)
    try:
        assert rt.twin_overlay.twin_scene == "mavis_v2_kitchen"
        assert rt.rail_sweep.scene_id == "mavis_v2"  # untouched
    finally:
        rt.gello.stop()
