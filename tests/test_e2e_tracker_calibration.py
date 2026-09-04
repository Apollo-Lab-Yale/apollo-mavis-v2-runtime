"""Phase-10 e2e: the tracker calibration wizard back end over a REAL uvicorn
server with the ``fake`` tracker backend — ``GET /api/tracker/calibration``
initial state, ``base_station start`` refused (409 "backend is not libsurvive"),
the yaw wizard driven end to end over REST (7 ``capture`` + ``apply``; telemetry
``tracker.settings.yaw_deg`` becomes the fit; a fresh Runtime over the same
``calibration_dir`` seeds ``yaw_deg`` from the persisted file), the telemetry
``tracker.calibration`` block, and the session/calibration mutual 409s."""

from __future__ import annotations

import json
import time

import httpx
import numpy as np
import pytest
from apollo_xarm7_core.protocol import TrackerCalibrationCommand
from conftest import LiveServer, make_runtime_config
from websockets.sync.client import connect as ws_connect

from apollo_xarm7_runtime.config import TrackerConfig
from apollo_xarm7_runtime.control.tracker_teleop import yaw_quat
from apollo_xarm7_runtime.runtime import Runtime

SPEC = {
    "mode": "teleop",
    "kind": "sim",
    "arms": ["view", "grip"],
    "frames": {"view": "arm_base:view", "grip": "arm_base:grip"},
    "sim_scene": "mavis_v2",
}
YAW_TRUE = 102.1
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


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

    def wait(self, pred, timeout_s=3.0) -> dict:
        deadline = time.monotonic() + timeout_s
        msg = self.latest()
        while not pred(msg) and time.monotonic() < deadline:
            msg = self.latest()
        return msg

    def close(self) -> None:
        self.sock.close()


def _post(api, kind, op, point=None):
    body = {"kind": kind, "op": op}
    if point is not None:
        body["point"] = point
    return api.post("/api/tracker/calibration", json=body)


def _gesture_raw() -> np.ndarray:
    """The lab gesture (left=+X, forward=-Y, 25 cm legs) as libsurvive sees it:
    ``p_world = Rz(YAW_TRUE) p_raw``."""
    from apollo_xarm7_core import se3

    leg = 0.25
    world = [np.array([0.3, 0.1, 0.2])]
    for d in ([leg, 0, 0], [0, -leg, 0], [-leg, 0, 0], [0, leg, 0], [0, 0, leg], [0, 0, -leg]):
        world.append(world[-1] + np.asarray(d, float))
    q = yaw_quat(-YAW_TRUE)
    return np.asarray([se3.quat_rotate(q, p) for p in world])


def _move_to(reader, frm, to, *, step_m=0.04, dwell=20):
    """Publish a sample path from ``frm`` to ``to`` (steps below max_jump_m so no
    sample is flagged as a jump), then ``dwell`` still samples at ``to`` (20 ms
    apart: longer than the 0.3 s capture averaging window)."""
    from apollo_xarm7_core import Pose

    n = max(1, int(np.ceil(np.linalg.norm(to - frm) / step_m)))
    for i in range(1, n + 1):
        p = frm + (to - frm) * (i / n)
        reader._publish(Pose(p, IDENT), np.zeros(3), np.zeros(3), time.monotonic())
        time.sleep(0.01)
    for _ in range(dwell):
        reader._publish(Pose(to, IDENT), np.zeros(3), np.zeros(3), time.monotonic())
        time.sleep(0.02)


def test_get_initial_status_and_telemetry_block(server, api):
    r = api.get("/api/tracker/calibration")
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["kind"] == "none" and st["phase"] == "idle" and st["detail"] == ""
    assert st["yaw_valid"] is True and st["yaw_calibrated_at"] is None
    assert st["base_station_installed_at"] is None
    assert st["lighthouses"] == [] and st["yaw_points"] == [] and st["fit_checks"] == []
    assert st["validation"] is None and st["next_point"] is None
    tele = Tele(server)
    try:
        trk = tele.latest()["tracker"]
        assert trk["backend"] == "fake"
        cal = trk["calibration"]
        assert cal is not None and cal["kind"] == "none" and cal["phase"] == "idle"
        assert cal["yaw_valid"] is True and cal["scenes"] == 0
    finally:
        tele.close()


def test_base_station_start_refused_on_fake_backend_and_bad_bodies(api):
    r = _post(api, "base_station", "start")
    assert r.status_code == 409 and r.json()["detail"] == "backend is not libsurvive", r.text
    r = _post(api, "base_station", "abort")
    assert r.status_code == 409
    assert r.json()["detail"] == "no base_station calibration in progress"
    for bad in ({"kind": "none", "op": "start"}, {"kind": "yaw", "op": "jump"}, {"op": "start"}):
        assert api.post("/api/tracker/calibration", json=bad).status_code == 422, bad
    assert api.get("/api/tracker/calibration").json()["phase"] == "idle"  # nothing started


def test_yaw_wizard_over_rest_applies_and_persists(server, api):
    reader = server.runtime.tracker
    raw = _gesture_raw()
    reader.stop()  # take over the fake sample stream: publish the gesture ourselves
    tele = Tele(server)
    try:
        r = _post(api, "yaw", "start")
        assert r.status_code == 200, r.text
        st = r.json()
        assert st["kind"] == "yaw" and st["phase"] == "capturing" and st["next_point"] == "start"
        cal = tele.wait(lambda m: m["tracker"]["calibration"]["phase"] == "capturing")
        assert cal["tracker"]["calibration"]["kind"] == "yaw"
        r = _post(api, "yaw", "apply")
        assert r.status_code == 409 and "capture all 7 points" in r.json()["detail"]
        prev = np.asarray(reader.status().pose_raw.position)  # the fake circle's last pose
        for i, label in enumerate(["start", "left", "forward", "right", "back", "up", "down"]):
            _move_to(reader, prev, raw[i])
            prev = raw[i]
            r = _post(api, "yaw", "capture", point=label if i % 2 else None)  # both spellings
            assert r.status_code == 200, (label, r.text)
            st = r.json()
            assert [p["label"] for p in st["yaw_points"]][-1] == label
            assert np.allclose(st["yaw_points"][-1]["pose"]["position"], raw[i], atol=1e-6)
        assert st["phase"] == "done" and st["next_point"] is None
        assert st["fitted_yaw_deg"] == pytest.approx(YAW_TRUE, abs=1e-3)
        assert st["fit_residual_deg"] == pytest.approx(0.0, abs=1e-3) and st["fit_checks"] == []
        r = _post(api, "yaw", "capture")
        assert r.status_code == 409 and "all 7 points captured" in r.json()["detail"]
        # Telemetry carries the fit before apply; settings unchanged so far.
        msg = tele.wait(lambda m: m["tracker"]["calibration"]["phase"] == "done")
        assert msg["tracker"]["calibration"]["fitted_yaw_deg"] == pytest.approx(YAW_TRUE, abs=1e-3)
        assert msg["tracker"]["settings"]["yaw_deg"] == 0.0
        r = _post(api, "yaw", "apply")
        assert r.status_code == 200, r.text
        st = r.json()
        assert st["applied_yaw_deg"] == pytest.approx(YAW_TRUE, abs=1e-3)
        assert st["yaw_valid"] is True and st["yaw_calibrated_at"] is not None
        msg = tele.wait(lambda m: m["tracker"]["settings"]["yaw_deg"] != 0.0)
        assert msg["tracker"]["settings"]["yaw_deg"] == pytest.approx(YAW_TRUE, abs=1e-3)
        assert msg["tracker"]["calibration"]["applied_yaw_deg"] == pytest.approx(YAW_TRUE, abs=1e-3)
        assert api.get("/api/tracker/calibration").json()["phase"] == "done"
        # Persisted; a "new process" over the same calibration_dir seeds yaw_deg from it.
        cfg = server.runtime.cfg
        persisted = json.loads((cfg.calibration_dir / "tracker_calibration.json").read_text())
        assert persisted["yaw_deg"] == pytest.approx(YAW_TRUE, abs=1e-3)
        assert persisted["yaw_valid"] is True
        cfg2 = cfg.model_copy(deep=True)
        cfg2.tracker.backend = "none"  # no second reader thread needed
        assert cfg2.tracker.yaw_deg == 0.0  # the YAML default ...
        rt2 = Runtime(cfg2)
        try:
            assert rt2.tracker_settings.get().yaw_deg == pytest.approx(YAW_TRUE, abs=1e-3)
            st2 = rt2.tracker_calibration.status()
            assert st2.yaw_calibrated_at == persisted["yaw_calibrated_at"]
        finally:
            rt2.stop()
    finally:
        tele.close()
        reader.start()  # hand the sample stream back to the fake backend


def test_session_and_calibration_refuse_each_other(server, api):
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    try:
        for _ in range(200):
            if api.get("/api/session").json()["state"] == "running":
                break
            time.sleep(0.05)
        r = _post(api, "yaw", "start")
        assert r.status_code == 409 and r.json()["detail"] == "stop the session first"
        r = _post(api, "base_station", "start")
        assert r.status_code == 409  # backend check first: still a clear refusal
    finally:
        api.delete("/api/session")
    r = _post(api, "yaw", "start")
    assert r.status_code == 200, r.text
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409 and r.json()["detail"] == "tracker calibration in progress"
    assert api.get("/api/session").status_code == 404
    r = _post(api, "yaw", "abort")
    assert r.status_code == 200 and r.json()["phase"] == "aborted"
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    api.delete("/api/session")


def test_post_session_gives_way_to_a_calibration_that_started_during_create(
    server, api, monkeypatch
):
    """``post_session`` checks ``active`` BEFORE ``manager.create`` and the
    calibration guard reads ``manager.session_active``, raised only inside
    ``create``: a ``start`` landing in between passes both guards. The re-check
    after ``create`` tears the fresh session down (409) and the calibration wins."""
    manager = server.runtime.manager
    real_create = manager.create
    armed = [True]

    def racing_create(spec):
        if armed.pop() if armed else False:  # one shot: the calibration slips in first
            st = server.runtime.tracker_calibration.command(
                TrackerCalibrationCommand(kind="yaw", op="start")
            )
            assert st.phase == "capturing"
        return real_create(spec)

    monkeypatch.setattr(manager, "create", racing_create)
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 409 and r.json()["detail"] == "tracker calibration in progress", r.text
    assert api.get("/api/session").status_code == 404
    assert manager.session_active is False
    st = api.get("/api/tracker/calibration").json()
    assert st["kind"] == "yaw" and st["phase"] == "capturing"  # untouched by the teardown
    r = api.post("/api/session", json=SPEC)  # the plain guard still holds
    assert r.status_code == 409 and r.json()["detail"] == "tracker calibration in progress"
    assert _post(api, "yaw", "abort").status_code == 200
    r = api.post("/api/session", json=SPEC)
    assert r.status_code == 200, r.text
    api.delete("/api/session")
